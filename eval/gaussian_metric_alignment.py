"""Compare retained-head Wasserstein, KR, and Gaussian Fisher metric tensors."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from train.experiment import ExperimentConfig, GaussianTransportNet, make_grids
from train.gaussian_projection import project_factor_directions, softplus_slope


BLOCKS = ("shape", "mean", "full")
MEASURES = ("raw_fisher_cosine", "projected_fisher_cosine", "raw_minus_projected")
STRATA = ("all", "condition_1_to_2", "condition_2_to_5", "condition_at_least_5")


def factor_gram(directions):
    features = directions.flatten(2)
    return features @ features.mT


@torch.no_grad()
def retained_metric_blocks(model, x):
    """Build compressed tensors for one output direction per head coordinate.

    The full head tensor for each block is the Kronecker product of the
    compressed tensor and the outer product of its augmented hidden vector.
    """
    hidden_mean = model.mean_net.get_feature_embeddings(x)
    hidden_shape = model.cov_net.get_feature_embeddings(x)
    raw = model.cov_net.head(hidden_shape)
    factor = model.cov_net(x)

    derivative = factor.new_zeros((len(x), 3, 2, 2))
    diagonal_slopes = softplus_slope(raw[:, (0, 2)])
    derivative[:, 0, 0, 0] = diagonal_slopes[:, 0]
    derivative[:, 1, 1, 0] = 1.0
    derivative[:, 2, 1, 1] = diagonal_slopes[:, 1]
    whitened_factor = torch.linalg.solve_triangular(factor[:, None], derivative, upper=False)
    fisher_field = (whitened_factor + whitened_factor.mT) / math.sqrt(2.0)
    projected_field = project_factor_directions(factor, derivative)
    identity = torch.eye(factor.shape[-1], dtype=factor.dtype, device=x.device).expand_as(factor)
    metrics = {
        "raw": (identity, factor_gram(derivative)),
        "projected": (identity, factor_gram(projected_field)),
        "fisher": (torch.cholesky_inverse(factor), factor_gram(fisher_field)),
    }
    singular_values = torch.linalg.svdvals(factor)
    condition_number = (singular_values[:, 0] / singular_values[:, -1]).square()
    kronecker_norms_sq = tuple(
        (hidden.square().sum(dim=1) + 1).square() for hidden in (hidden_mean, hidden_shape)
    )
    return metrics, kronecker_norms_sq, condition_number


def alignment(metric_blocks, kronecker_norms_sq):
    """Frobenius cosines of full parameter-metric blocks, computed compactly."""
    fisher_mean, fisher_shape = metric_blocks["fisher"]
    mean_weight, shape_weight = kronecker_norms_sq
    comparisons = {}
    for method in ("raw", "projected"):
        mean, shape = metric_blocks[method]
        cross_mean = (mean * fisher_mean).sum(dim=(1, 2)) * mean_weight
        cross_shape = (shape * fisher_shape).sum(dim=(1, 2)) * shape_weight
        method_mean_sq = mean.square().sum(dim=(1, 2)) * mean_weight
        method_shape_sq = shape.square().sum(dim=(1, 2)) * shape_weight
        fisher_mean_sq = fisher_mean.square().sum(dim=(1, 2)) * mean_weight
        fisher_shape_sq = fisher_shape.square().sum(dim=(1, 2)) * shape_weight
        comparisons[method] = {
            "mean": cross_mean / (method_mean_sq * fisher_mean_sq).sqrt(),
            "shape": cross_shape / (method_shape_sq * fisher_shape_sq).sqrt(),
            "full": (cross_mean + cross_shape) / (
                (method_mean_sq + method_shape_sq) * (fisher_mean_sq + fisher_shape_sq)
            ).sqrt(),
        }
    return {
        block: torch.stack((
            comparisons["raw"][block], comparisons["projected"][block],
            comparisons["raw"][block] - comparisons["projected"][block],
        ), dim=1).numpy()
        for block in BLOCKS
    }


def load_seed(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = ExperimentConfig(**saved["config"])
    x, _, _ = make_grids(config, torch.device("cpu"))
    model = GaussianTransportNet(config.hidden_dim).double().eval()
    checkpoints = [
        row for row in saved["checkpoints"]
        if (row["round"] == 0 and row["trajectory"] == "shared")
        or (row["round"] > 0 and row["trajectory"] == "transport")
    ]
    rounds = [row["round"] for row in checkpoints]
    if rounds != list(range(config.n_rounds + 1)):
        raise ValueError(f"{path} must contain the complete original tNTK trajectory")
    values = []
    for checkpoint in checkpoints:
        model.load_state_dict(checkpoint["state_dict"])
        tensors, weights, condition = retained_metric_blocks(model, x.double())
        scores = alignment(tensors, weights)
        if not torch.isfinite(condition).all() or not all(np.isfinite(a).all() for a in scores.values()):
            raise ValueError(f"Nonfinite Fisher alignment for {path} at round {checkpoint['round']}")
        values.append((condition.numpy(), scores))
    return int(saved["seed"]), values


def stratify(condition):
    return dict(zip(STRATA, (
        np.ones_like(condition, dtype=bool), condition < 2,
        (condition >= 2) & (condition < 5), condition >= 5,
    ), strict=True))


def aggregate_seed(seed, checkpoints):
    scopes = [("all", checkpoints), *(
        (str(i), [checkpoint]) for i, checkpoint in enumerate(checkpoints)
    )]
    for round_selector, selected_checkpoints in scopes:
        condition = np.concatenate([values[0] for values in selected_checkpoints])
        for stratum, mask in stratify(condition).items():
            if not mask.any():
                continue
            for block in BLOCKS:
                values = np.concatenate([row[1][block] for row in selected_checkpoints])
                yield {
                    "seed": seed, "round": round_selector, "stratum": stratum, "block": block,
                    "n_points": int(mask.sum()), "mean_condition_number": float(condition[mask].mean()),
                    **dict(zip(MEASURES, values[mask].mean(axis=0).tolist(), strict=True)),
                }


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["round"], row["stratum"], row["block"])
        groups.setdefault(key, []).append(row)
    rng = np.random.default_rng(3216)
    summary = []
    for (round_selector, stratum, block), seed_rows in groups.items():
        stats = {"round": round_selector, "stratum": stratum, "block": block,
                 "n_seeds": len(seed_rows), "n_points": sum(row["n_points"] for row in seed_rows)}
        for name in (*MEASURES, "mean_condition_number"):
            values = np.array([row[name] for row in seed_rows])
            stats[name] = float(values.mean())
        differences = np.array([row["raw_minus_projected"] for row in seed_rows])
        uncertainty = None
        if len(differences) > 1:
            draw = rng.integers(len(differences), size=(10_000, len(differences)))
            uncertainty = np.quantile(differences[draw].mean(axis=1), [0.025, 0.975]).tolist()
        stats["raw_minus_projected_ci95"] = uncertainty
        summary.append(stats)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.checkpoints.glob("seed_*.pt"))
    if not paths:
        parser.error("Expected replayed seed_*.pt files in --checkpoints")

    torch.set_num_threads(1)
    rows, seeds = [], set()
    for path in paths:
        seed, values = load_seed(path)
        if seed in seeds:
            raise ValueError(f"Duplicate replay seed {seed}")
        seeds.add(seed)
        rows.extend(aggregate_seed(seed, values))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "seeds": sorted(seeds), "trajectory": "original tNTK, with shared round zero",
        "estimand": "Mean seed-level Frobenius cosine over checkpoint and pool points",
        "bootstrap": "Percentile 95% interval, 10,000 paired seed resamples",
        "rows": summarize(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Compared retained-head metrics for {len(seeds)} replay seeds")


if __name__ == "__main__":
    main()
