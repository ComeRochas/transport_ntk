"""Measure kernel and acquisition stability under latent rotations at round zero."""

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

from train.experiment import (
    GaussianTransportNet,
    compute_transport_kernel,
    get_distribution_parameters,
    load_config,
    make_grids,
    select_batch_min_total_var,
    train_model,
)
from train.gaussian_projection import (
    compute_projected_transport_kernel,
    project_factor_directions,
    softplus_slope,
)


METHODS = ("transport", "projected", "projected_canonical")
ROTATIONS = (
    ("constant_pi_over_3", math.pi / 3, True),
    ("local_pi_over_4", math.pi / 4, False),
    ("local_pi", math.pi, False),
    ("local_2pi", 2 * math.pi, False),
)
METRICS = (
    "relative_covariance_error", "relative_kernel_frobenius", "relative_shape_frobenius",
    "max_relative_diagonal_error", "batch_overlap", "same_batch_order", "first_score_top10_overlap",
)


def latent_rotation(x, amplitude, *, constant=False):
    """Return Q(x) with angle a·(x₁ − x₂ + x₁x₂), or a global angle a."""
    x1, x2 = x.unbind(dim=1)
    angle = torch.full_like(x1, amplitude) if constant else amplitude * (x1 - x2 + x1 * x2)
    c, s = angle.cos(), angle.sin()
    return torch.stack((c, -s, s, c), dim=1).reshape(-1, 2, 2)


@torch.no_grad()
def rotated_kernels(model, x, rotation):
    """Return full and covariance-only Gram matrices for the actual rotated map.

    Factor directions are compressed by output coordinate. A direction wrt a
    last-layer weight also carries the corresponding hidden activation. Bias
    directions carry 1. The rotation is prescribed and independent of weights.
    """
    mean_hidden = model.mean_net.get_feature_embeddings(x)
    cov_hidden = model.cov_net.get_feature_embeddings(x)
    raw = model.cov_net.head(cov_hidden)
    factor = model.cov_net(x)

    diagonal_slopes = softplus_slope(raw[:, (0, 2)])
    directions = factor.new_zeros((len(x), 3, 2, 2))
    directions[:, 0, 0, 0] = diagonal_slopes[:, 0]
    directions[:, 1, 1, 0] = 1.0
    directions[:, 2, 1, 1] = diagonal_slopes[:, 1]

    rotated_factor = factor @ rotation
    rotated_directions = directions @ rotation[:, None]
    representations = {
        "transport": rotated_directions,
        "projected": project_factor_directions(rotated_factor, rotated_directions),
        "projected_canonical": project_factor_directions(
            rotated_factor, rotated_directions, canonical=True,
        ),
    }
    mean_gram = 2.0 * (mean_hidden @ mean_hidden.T + 1.0)
    cov_weights = cov_hidden @ cov_hidden.T + 1.0
    kernels = {}
    for method, representation in representations.items():
        flat = representation.flatten(1)
        shape_gram = cov_weights * (flat @ flat.T)
        kernels[method] = (mean_gram + shape_gram, shape_gram)
    return rotated_factor, kernels


def first_score_top10(gram, observed, lam):
    """Rank the eligible pool points using the original first BAIT score."""
    pool = torch.arange(len(gram), device=gram.device)
    observed = torch.tensor(observed, dtype=torch.long, device=gram.device)
    cross = gram[:, observed]
    observed_gram = gram[observed][:, observed] + lam * torch.eye(
        len(observed), device=gram.device, dtype=gram.dtype,
    )
    lower = torch.linalg.cholesky(observed_gram)
    factors = torch.linalg.solve_triangular(lower, cross.T, upper=False)
    posterior = gram - factors.T @ factors
    scores = posterior.square().sum(dim=0) / (posterior.diag() + 1.0)
    remaining = pool[~torch.isin(pool, observed)]
    ordering = scores[remaining].argsort(descending=True, stable=True)
    return remaining[ordering[:10]].tolist()


def relative_frobenius(current, reference):
    return float(torch.linalg.vector_norm(current - reference) / torch.linalg.vector_norm(reference))


def run_seed(seed, config):
    torch.manual_seed(seed)
    x, _, observed = make_grids(config, torch.device("cpu"))
    x_observed = x[observed]
    mean, factor = get_distribution_parameters(x_observed, config)
    model = GaussianTransportNet(config.hidden_dim)
    train_model(model, x_observed, mean, factor, config, verbose=False)
    # Keep the fitted weights and perform the invariance comparison in float64.
    model.double().eval()
    x = x.double()

    reference_factor, reference_kernels = rotated_kernels(model, x, latent_rotation(x, 0.0))
    reference_covariance = reference_factor @ reference_factor.mT
    reference_impl = {
        "transport": compute_transport_kernel(model, x),
        "projected": compute_projected_transport_kernel(model, x),
        "projected_canonical": compute_projected_transport_kernel(model, x, canonical=True),
    }
    identity_error = max(
        relative_frobenius(reference_kernels[arm][0], reference_impl[arm]) for arm in METHODS
    )
    if identity_error > 1e-9:
        raise AssertionError(f"Unrotated Jacobian Gram differs from the benchmark for seed {seed}")

    pool = list(range(len(x)))
    reference_choices = {
        arm: (
            select_batch_min_total_var(k, pool, observed, config.batch_size, config.lam),
            first_score_top10(k, observed, config.lam),
        )
        for arm, (k, _) in reference_kernels.items()
    }
    rows = []
    for name, angle, constant in ROTATIONS:
        rotated_factor, kernels = rotated_kernels(model, x, latent_rotation(x, angle, constant=constant))
        covariance_error = relative_frobenius(rotated_factor @ rotated_factor.mT, reference_covariance)
        for arm, (gram, shape_gram) in kernels.items():
            ref_gram, ref_shape = reference_kernels[arm]
            reference_batch, reference_top10 = reference_choices[arm]
            batch = select_batch_min_total_var(gram, pool, observed, config.batch_size, config.lam)
            top10 = first_score_top10(gram, observed, config.lam)
            rows.append({
                "seed": seed, "rotation": name, "method": arm,
                "angle_amplitude_rad": angle, "relative_covariance_error": covariance_error,
                "relative_kernel_frobenius": relative_frobenius(gram, ref_gram),
                "relative_shape_frobenius": relative_frobenius(shape_gram, ref_shape),
                "max_relative_diagonal_error": float(((gram.diag() - ref_gram.diag()) / ref_gram.diag()).abs().max()),
                "batch_overlap": len(set(batch) & set(reference_batch)) / len(reference_batch),
                "same_batch_order": int(batch == reference_batch),
                "first_score_top10_overlap": len(set(top10) & set(reference_top10)) / len(reference_top10),
                "reference_batch": json.dumps(reference_batch), "rotated_batch": json.dumps(batch),
                "reference_first_score_top10": json.dumps(reference_top10),
                "rotated_first_score_top10": json.dumps(top10),
            })
    return identity_error, rows


def summarize(rows):
    summary = {}
    for name, _, _ in ROTATIONS:
        summary[name] = {}
        for arm in METHODS:
            selected = [row for row in rows if row["rotation"] == name and row["method"] == arm]
            summary[name][arm] = {}
            for metric in METRICS:
                values = np.array([row[metric] for row in selected])
                summary[name][arm][metric] = {
                    "mean": float(values.mean()),
                    "se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
                    "max": float(values.max()),
                }
    return summary


def draw(fig_path, summary):
    rotation_names = [name for name, _, _ in ROTATIONS]
    positions = np.arange(len(ROTATIONS))
    labels = ["Global π/3", "Local π/4", "Local π", "Local 2π"]
    styles = [("tNTK", "#2c5d91", "o"), ("Projected pullback", "#22785b", "s"),
              ("Projected canonical", "#7b5396", "^")]
    panels = [("relative_kernel_frobenius", "Relative Gram change"),
              ("batch_overlap", "First batch overlap"),
              ("first_score_top10_overlap", "First score top-10 overlap")]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.65), layout="constrained")
    for axis, (metric, title) in zip(axes, panels):
        for arm, (label, color, marker), offset in zip(METHODS, styles, (-0.13, 0, 0.13)):
            values = [summary[name][arm][metric] for name in rotation_names]
            means = [entry["mean"] for entry in values]
            errors = [entry["se"] or 0.0 for entry in values]
            axis.errorbar(positions + offset, means, yerr=errors, fmt=marker, color=color, label=label)
        if metric != "relative_kernel_frobenius":
            axis.set_ylim(-0.05, 1.10)
        axis.set(title=title, xticks=positions, xticklabels=labels)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["right", "top"]].set_visible(False)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False)
    figure.savefig(fig_path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    torch.set_num_threads(1)
    config = replace(load_config("both_complex_b2"), device="cpu")

    rows = []
    identity_errors = []
    for seed in range(args.seeds):
        identity_error, seed_rows = run_seed(seed, config)
        identity_errors.append(identity_error)
        rows.extend(seed_rows)
        print(f"Round-zero fit and rotation checks complete for seed {seed}", flush=True)
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_seed.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "n_seeds": args.seeds, "training_steps": config.n_train_steps, "pool_size": 196,
        "batch_size": config.batch_size, "conditioned_noise": config.lam, "hypothetical_noise": 1.0,
        "kernel_evaluation_dtype": "float64",
        "max_unrotated_error_vs_benchmark_kernel": max(identity_errors), "rotations": summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    draw(args.output_dir / "gauge_invariance.png", summary)


if __name__ == "__main__":
    main()
