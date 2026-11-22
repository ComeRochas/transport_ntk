"""Compare GP arithmetic and kernel precision on replayed Gaussian checkpoints."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from train.experiment import ExperimentConfig, GaussianTransportNet, make_grids, select_batch_min_total_var
from train.gaussian_gp import acquisition_model, posterior_covariance, variance_reduction_scores
from train.gaussian_validation import KERNEL_ARMS, PROTOCOLS, gaussian_kernels


MODES = ("native32", "legacy64", "rounded32_solve64", "native64_solve")
REFERENCE = MODES[-1]
NEAR_TIE = 1e-4


def legacy_posterior(kernel, labels, noise):
    idx = torch.as_tensor(labels, dtype=torch.long)
    cross = kernel[:, idx]
    block = kernel[idx][:, idx] + noise * torch.eye(len(labels), dtype=kernel.dtype)
    inverse = torch.linalg.inv(torch.linalg.cholesky(block))
    return kernel - cross @ (inverse.T @ inverse) @ cross.T


def trace(kernel, labels, batch_size, observed, hypothetical, *, legacy=False, follow=None):
    posterior = legacy_posterior(kernel, labels, observed) if legacy else None
    selected = list(labels)
    scores_and_posteriors = []
    for step in range(batch_size):
        if not legacy:
            posterior = posterior_covariance(
                kernel, selected, n_observed=len(labels), observed_noise=observed,
                hypothetical_noise=hypothetical,
            )
        scores = variance_reduction_scores(posterior, hypothetical)
        eligible = torch.tensor(sorted(set(range(len(kernel))) - set(selected)), dtype=torch.long)
        if not len(eligible):
            break
        rank = eligible[torch.argsort(scores[eligible], descending=True, stable=True)]
        scores_and_posteriors.append((rank, scores, posterior))
        pick = rank[0].item() if follow is None else follow[step]
        selected.append(pick)
        if legacy:
            column = posterior[:, pick]
            posterior = posterior - torch.outer(column, column) / (posterior[pick, pick] + hypothetical)
    return selected[len(labels):], scores_and_posteriors


def spectral_extrema(matrix):
    matrix = matrix.double()
    lo, hi = torch.linalg.eigvalsh((matrix + matrix.T) * 0.5)[[0, -1]].tolist()
    return lo, lo / hi if hi > 0 else np.nan


def rank_statistics(rank, reference):
    index = torch.empty(int(torch.maximum(rank.max(), reference.max())) + 1, dtype=torch.int64)
    index[reference] = torch.arange(len(reference), dtype=torch.int64)
    diff = index[rank] - torch.arange(len(rank), dtype=torch.int64)
    spearman = 1.0 if len(rank) < 2 else 1.0 - 6.0 * diff.square().sum().item() / (len(rank) * (len(rank)**2 - 1))
    top = min(10, len(rank))
    return spearman, len(set(rank[:top].tolist()) & set(reference[:top].tolist())) / top


def audit_case(kernel32, kernel64, labels, batch_size, noise, hypothetical, *, check_legacy=True):
    grams = dict(native32=kernel32, legacy64=kernel64,
                 rounded32_solve64=kernel32.double(), native64_solve=kernel64)
    runs = {
        mode: trace(gram, labels, batch_size, noise, hypothetical, legacy=mode in MODES[:2])
        for mode, gram in grams.items()
    }
    reference_batch, reference_steps = runs[REFERENCE]
    if check_legacy:
        expected = select_batch_min_total_var(
            kernel32, range(len(kernel32)), labels, batch_size, noise, hypothetical_lam=hypothetical,
        )
        if runs["native32"][0] != expected:
            raise RuntimeError("Audit trace did not reproduce published selector")

    rows = []
    for mode, (batch, steps) in runs.items():
        aligned = steps
        if batch[:-1] != reference_batch[:-1]:
            _, aligned = trace(grams[mode], labels, batch_size, noise, hypothetical,
                               legacy=mode in MODES[:2], follow=reference_batch)
        for step, ((ref_rank, ref_scores, _), (rank, scores, _), (own_rank, _, posterior)) in enumerate(
            zip(reference_steps, aligned, steps, strict=True), start=1,
        ):
            winner, next_best = ref_rank[:2].tolist()
            test_winner = rank[0].item()
            best = ref_scores[winner].item()
            second = ref_scores[next_best].item()
            denominator = max(abs(best), np.finfo(float).tiny)
            gap = (best - second) / denominator
            eigen, eigen_relative = spectral_extrema(posterior)
            spearman, top10 = rank_statistics(rank, ref_rank)
            rows.append({
                "mode": mode, "step": step, "picked": own_rank[0].item(),
                "ref_picked": winner, "same_prefix_pick": test_winner,
                "same_prefix_first_match": int(test_winner == winner),
                "same_prefix_spearman": spearman, "same_prefix_top10": top10,
                "ref_best_score": best, "ref_second_score": second, "reference_relative_gap": gap,
                "ref_near_tie_count": int(((best - ref_scores[ref_rank]) / denominator <= NEAR_TIE).sum()),
                "test_best_score": scores[test_winner].item(),
                "same_prefix_relative_regret": (best - ref_scores[test_winner].item()) / denominator,
                "min_own_posterior_eigenvalue": eigen, "min_own_posterior_eigenvalue_relative": eigen_relative,
                "ordered_batch_match": int(batch == reference_batch),
                "batch_overlap": len(set(batch) & set(reference_batch)) / batch_size,
                "batch": json.dumps(batch), "reference_batch": json.dumps(reference_batch),
            })
    return rows


def rows_for_seed(path, include_final):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = ExperimentConfig(**payload["config"])
    x_pool, _, _ = make_grids(config, torch.device("cpu"))
    model = GaussianTransportNet(config.hidden_dim).eval()
    for checkpoint in payload["checkpoints"]:
        round_index = checkpoint["round"]
        if round_index == config.n_rounds and not include_final:
            continue
        model.load_state_dict(checkpoint["state_dict"])
        model64, inputs64 = acquisition_model(model, x_pool)
        kernel32 = gaussian_kernels(model, x_pool)
        kernel64 = gaussian_kernels(model64, inputs64)
        labels = checkpoint["ordered_labels"]
        for arm in KERNEL_ARMS:
            raw32, raw64 = kernel32[arm], kernel64[arm]
            gram_error = (torch.linalg.vector_norm(raw32.double() - raw64)
                          / torch.linalg.vector_norm(raw64)).item()
            for protocol, (normalize, coherent) in PROTOCOLS.items():
                gram32 = raw32 / raw32.diagonal().mean() if normalize else raw32
                gram64 = raw64 / raw64.diagonal().mean() if normalize else raw64
                rows = audit_case(gram32, gram64, labels, config.batch_size, config.lam,
                                  config.lam if coherent else 1.0)
                if protocol == "original":
                    history = checkpoint["historical_next_batches"].get(arm)
                    if history and history != json.loads(rows[0]["batch"]):
                        raise RuntimeError(f"Historical choice mismatch for seed {payload['seed']}")
                for row in rows:
                    yield {
                        "seed": payload["seed"], "round": round_index,
                        "checkpoint_policy": checkpoint["trajectory"], "arm": arm,
                        "protocol": protocol, "raw_relative_gram_error": gram_error, **row,
                    }


def summarize(rows):
    summary = {}
    for protocol in PROTOCOLS:
        cases = [r for r in rows if r["protocol"] == protocol]
        by_arm = {}
        for arm in ("all", *KERNEL_ARMS):
            active = cases if arm == "all" else [r for r in cases if r["arm"] == arm]
            by_mode = {}
            for mode in MODES:
                steps = [r for r in active if r["mode"] == mode]
                first = [r for r in steps if r["step"] == 1]
                switched = [r for r in steps if not r["same_prefix_first_match"]]
                eigen = np.array([r["min_own_posterior_eigenvalue"] for r in steps])
                rel_eigen = np.array([r["min_own_posterior_eigenvalue_relative"] for r in steps])
                gaps = np.array([r["reference_relative_gap"] for r in steps])
                by_mode[mode] = {
                    "n_batches": len(first),
                    "first_choice_changes": sum(not r["same_prefix_first_match"] for r in first),
                    "ordered_batch_changes": sum(not r["ordered_batch_match"] for r in first),
                    "unordered_batch_changes": sum(r["batch_overlap"] < 1 for r in first),
                    "mean_same_prefix_spearman": float(np.mean([r["same_prefix_spearman"] for r in steps])),
                    "mean_same_prefix_top10_overlap": float(np.mean([r["same_prefix_top10"] for r in steps])),
                    "min_posterior_eigenvalue": float(eigen.min()),
                    "median_min_posterior_eigenvalue": float(np.median(eigen)),
                    "min_relative_posterior_eigenvalue": float(rel_eigen.min()),
                    "posterior_min_below_minus_1e_6_count": int((eigen < -1e-6).sum()),
                    "reference_gap_median": float(np.median(gaps)),
                    "reference_gap_below_1e_4_count": int((gaps <= NEAR_TIE).sum()),
                    "same_prefix_winner_changes": len(switched),
                    "same_prefix_changes_outside_1e_4": sum(
                        r["same_prefix_relative_regret"] > NEAR_TIE for r in switched
                    ),
                    "max_ref_relative_regret": max((r["same_prefix_relative_regret"] for r in switched), default=0.0),
                }
            by_arm[arm] = by_mode
        summary[protocol] = by_arm
    return {
        "reference": REFERENCE, "near_tie_relative_score_threshold": NEAR_TIE,
        "seeds": sorted({r["seed"] for r in rows}),
        "checkpoints": len({(r["seed"], r["round"], r["checkpoint_policy"]) for r in rows}),
        "protocols": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-final", action="store_true", help="Include an extra hypothetical query after round 5")
    parser.add_argument("--limit-seeds", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    files = sorted(args.checkpoint_dir.glob("seed_*.pt"))[:args.limit_seeds]
    if not files:
        parser.error("No checkpoint files found")
    rows = []
    for path in files:
        rows.extend(rows_for_seed(path, args.include_final))
        print(f"Audited {path.stem}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output_dir / "precision_steps.csv.gz", "wt", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summarize(rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
