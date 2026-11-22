"""Sweep the hypothetical GP noise on fixed Gaussian benchmark checkpoints."""

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy.stats import rankdata

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

from eval.gaussian_energy_analysis import paired_bootstrap
from train.experiment import ExperimentConfig, GaussianTransportNet, make_grids
from train.gaussian_validation import gaussian_kernels


ARMS = ("transport", "projected", "projected_canonical")
SCALES = ("raw", "unit_mean_diag")
OBSERVED_NOISE = 0.001
HISTORICAL_NOISE = 1.0
NOISES = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
LABELS = {"transport": "tNTK", "projected": "Pullback", "projected_canonical": "Canonical"}
COLORS = {"transport": "#245c91", "projected": "#197555", "projected_canonical": "#8152a4"}
SCALE_LABELS = {"raw": "Raw Gram", "unit_mean_diag": "Unit mean diagonal"}


def condition(K, labels, noise):
    observed = K[np.ix_(labels, labels)] + noise * np.eye(len(labels))
    chol = np.linalg.cholesky(observed)
    whitened = solve_triangular(chol, K[labels, :], lower=True)
    residual = K - whitened.T @ whitened
    return (residual + residual.T) / 2, observed


def greedy_batch(posterior, remaining, noise, batch_size):
    """Use a rank-one posterior update after each hypothetical query."""
    current = posterior
    eligible = remaining.copy()
    batch = []
    first_scores = None
    for step in range(batch_size):
        columns = current[:, eligible]
        denominator = current[eligible, eligible] + noise
        gains = np.square(columns).sum(axis=0) / denominator
        if np.any(denominator <= 0) or not np.isfinite(gains).all():
            raise ValueError("Nonfinite or nonpositive GP score denominator")
        if step == 0:
            first_scores = gains
        best = int(gains.argmax())
        j = int(eligible[best])
        batch.append(j)
        if step + 1 < batch_size:
            column = columns[:, best]
            current = current - np.outer(column, column) / denominator[best]
            eligible = np.delete(eligible, best)
    return first_scores, tuple(batch)


def overlap(a, b):
    return len(set(a) & set(b)) / len(a)


def ranking_comparison(a, b):
    scores_a, batch_a = a
    scores_b, batch_b = b
    rank_a, rank_b = rankdata(scores_a), rankdata(scores_b)
    spearman = None if np.ptp(rank_a) == 0 or np.ptp(rank_b) == 0 else float(
        np.clip(np.corrcoef(rank_a, rank_b)[0, 1], -1, 1)
    )
    top = min(10, len(scores_a))
    top_a = np.argsort(-scores_a, kind="stable")[:top]
    top_b = np.argsort(-scores_b, kind="stable")[:top]
    return dict(
        rank_spearman=spearman, top10_overlap=overlap(top_a, top_b),
        first_match=int(batch_a[0] == batch_b[0]), batch_overlap=overlap(batch_a, batch_b),
        exact_ordered_batch_match=int(batch_a == batch_b),
    )


def spectrum(K):
    eig = np.linalg.eigvalsh(K)
    mass = np.maximum(eig[::-1], 0)
    mass /= mass.sum()
    return mass, dict(
        min_eigenvalue_over_trace=float(eig[0] / np.trace(K)),
        effective_rank=float(1 / np.square(mass).sum()),
        rank_99pct=int(np.searchsorted(mass.cumsum(), 0.99) + 1),
        largest_eigenvalue_share=float(mass[0]),
    )


def relative_frobenius(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def correlations(K):
    scale = np.sqrt(K.diagonal())
    return K / np.outer(scale, scale)


def check_scale_equivariance(K, scale, labels, eligible, batch_size, raw_result):
    posterior, _ = condition(K / scale, labels, OBSERVED_NOISE / scale)
    rescaled = greedy_batch(posterior, eligible, HISTORICAL_NOISE / scale, batch_size)
    return {
        "rescale_both_noises_relative_first_score_error": relative_frobenius(raw_result[0], scale * rescaled[0]),
        "rescale_both_noises_batch_overlap": overlap(raw_result[1], rescaled[1]),
    }


def replay_grams(directory, trajectory):
    files = sorted(directory.glob("seed_*.pt"))
    if not files:
        raise ValueError(f"No seed_*.pt checkpoints in {directory}")
    seen = set()
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        config = ExperimentConfig(**payload["config"])
        if config.lam != OBSERVED_NOISE:
            raise ValueError("The replay uses a different historical observation noise")
        grid, _, _ = make_grids(config, torch.device("cpu"))
        grid = grid.to(dtype=torch.float64)
        model = GaussianTransportNet(config.hidden_dim).double().eval()
        records = [r for r in payload["checkpoints"] if r["trajectory"] in ("shared", trajectory)]
        if sorted(r["round"] for r in records) != list(range(config.n_rounds + 1)):
            raise ValueError(f"Incomplete replay for seed {payload['seed']} and {trajectory}")
        for record in records:
            key = (payload["seed"], record["round"])
            if key in seen:
                raise ValueError(f"Duplicate checkpoint {key}")
            seen.add(key)
            model.load_state_dict(record["state_dict"])
            with torch.no_grad():
                kernels = gaussian_kernels(model, grid)
            grams = {a: kernels[a].numpy() for a in ARMS}
            yield key, record["trajectory"], record["ordered_labels"], config.batch_size, grams


def analyze_checkpoint(key, trajectory, labels, batch_size, grams, spectra):
    seed, round_index = key
    meta = dict(seed=seed, round=round_index, trajectory=trajectory)
    pool = np.arange(len(grams[ARMS[0]]))
    eligible = np.setdiff1d(pool, labels)
    noise_rows, structure_rows, comparison_rows, pair_rows = [], [], [], []
    selection = {}
    normalizers = {arm: float(grams[arm].diagonal().mean()) for arm in ARMS}
    for arm, K in grams.items():
        mean_diag = normalizers[arm]
        prior_mass, prior_spectrum = spectrum(K)
        spectra["prior", arm].append(prior_mass)
        for scale in SCALES:
            scaled = K if scale == "raw" else K / mean_diag
            posterior, observed = condition(scaled, labels, OBSERVED_NOISE)
            post_mass, post_spectrum = spectrum(posterior)
            spectra[scale, arm].append(post_mass)
            observed_eigs = np.linalg.eigvalsh(observed)
            candidate_var = posterior[eligible, eligible]
            baseline, coherent = HISTORICAL_NOISE, OBSERVED_NOISE
            results = {noise: greedy_batch(posterior, eligible, noise, batch_size) for noise in NOISES}
            selection[scale, arm] = results
            invariant = check_scale_equivariance(
                K, mean_diag, labels, eligible, batch_size, results[baseline]
            ) if scale == "raw" else {}
            structure = dict(
                **meta, scale=scale, arm=arm, raw_mean_diagonal=mean_diag,
                prior_diag_cv=float(K.diagonal().std() / mean_diag),
                prior_diag_p10_relative_to_mean=float(np.quantile(K.diagonal() / mean_diag, 0.10)),
                prior_diag_p90_relative_to_mean=float(np.quantile(K.diagonal() / mean_diag, 0.90)),
                observed_condition_number=float(observed_eigs[-1] / observed_eigs[0]),
                posterior_candidate_variance_median=float(np.median(candidate_var)),
                posterior_candidate_variance_p10=float(np.quantile(candidate_var, 0.10)),
                posterior_candidate_variance_p90=float(np.quantile(candidate_var, 0.90)),
                posterior_candidate_variance_median_relative_to_prior_mean=float(
                    np.median(candidate_var) / np.mean(scaled.diagonal())
                ),
                **{f"prior_{k}": v for k, v in prior_spectrum.items()},
                **{f"posterior_{k}": v for k, v in post_spectrum.items()},
            )
            structure.update(dict.fromkeys(
                ("rescale_both_noises_relative_first_score_error", "rescale_both_noises_batch_overlap")
            ) | invariant)
            structure_rows.append(structure)
            for noise, (scores, batch) in results.items():
                historical_cmp = ranking_comparison(results[baseline], (scores, batch))
                coherent_cmp = ranking_comparison(results[coherent], (scores, batch))
                runner_up, best = np.sort(np.partition(scores, -2)[-2:])
                noise_rows.append(dict(
                    **meta, scale=scale, arm=arm, hypothetical_noise=noise,
                    batch=json.dumps(batch), first_index=batch[0],
                    eligible_count=len(eligible),
                    first_score_relative_margin=float((best - runner_up) / best) if best else 0.0,
                    hypothetical_noise_over_median_candidate_variance=float(noise / np.median(candidate_var)),
                    share_candidate_variance_le_hypothetical_noise=float(np.mean(candidate_var <= noise)),
                    **{f"{k}_vs_noise_1": v for k, v in historical_cmp.items()},
                    **{f"{k}_vs_noise_0p001": v for k, v in coherent_cmp.items()},
                ))
    for left, right in combinations(ARMS, 2):
        base, other = grams[left], grams[right]
        pair_rows.append(dict(
            **meta, left=left, right=right,
            ratio_raw_mean_diagonal=float(normalizers[right] / normalizers[left]),
            nominal_relative_frobenius=relative_frobenius(base, other),
            unit_mean_diagonal_relative_frobenius=relative_frobenius(
                base / normalizers[left], other / normalizers[right]
            ),
            correlation_relative_frobenius=relative_frobenius(correlations(base), correlations(other)),
            relative_max_diagonal_difference=float(
                np.max(np.abs(base.diagonal() - other.diagonal())) / normalizers[left]
            ),
        ))
        for scale in SCALES:
            first, second = (selection[scale, variant] for variant in (left, right))
            for noise in NOISES:
                comparison_rows.append(dict(
                    **meta, comparison="across_kernels", scale=scale, left=left, right=right,
                    hypothetical_noise=noise, **ranking_comparison(first[noise], second[noise]),
                ))
    for arm in ARMS:
        for noise in NOISES:
            comparison_rows.append(dict(
                **meta, comparison="across_scales", scale="raw_vs_unit_mean_diag", left=arm, right=arm,
                hypothetical_noise=noise,
                **ranking_comparison(*(selection[scale, arm][noise] for scale in SCALES)),
            ))
    return noise_rows, structure_rows, comparison_rows, pair_rows


def summarize_groups(rows, group_fields, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_fields)].append(row)
    return [
        dict(zip(group_fields, key)) | {metric: paired_bootstrap(group, metric) for metric in metrics}
        for key, group in groups.items()
    ]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_sweep(path, rows):
    metrics = (
        ("rank_spearman_vs_noise_1", "First-step rank correlation"),
        ("first_match_vs_noise_1", "Same first acquisition"),
        ("batch_overlap_vs_noise_1", "Batch overlap"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.8), layout="constrained", sharex=True)
    for i, scale in enumerate(SCALES):
        for j, (metric, title) in enumerate(metrics):
            ax = axes[i, j]
            for arm in ARMS:
                data = [r for r in rows if r["scale"] == scale and r["arm"] == arm]
                noises = np.array([r["hypothetical_noise"] for r in data])
                values = [r[metric] for r in data]
                ax.semilogx(noises, [v["mean"] for v in values], "o-", color=COLORS[arm],
                            label=LABELS[arm], markersize=3)
                if values[0]["ci95"] is not None:
                    low, high = np.array([v["ci95"] for v in values]).T
                    ax.fill_between(noises, low, high, color=COLORS[arm], alpha=0.13)
            ax.axvline(OBSERVED_NOISE, ls="--", color="#777777", lw=0.9)
            ax.set(ylim=(-0.04, 1.05))
            ax.grid(axis="y", alpha=0.21)
            ax.spines[["right", "top"]].set_visible(False)
            if j:
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(SCALE_LABELS[scale])
            if i == 1:
                ax.set_xlabel("Hypothetical noise")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_spectra(path, spectra):
    labels = (("prior", "Prior Gram"), ("raw", "Posterior, raw Gram"),
              ("unit_mean_diag", "Posterior, unit mean diagonal"))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.35), layout="constrained", sharey=True)
    for (key, title), ax in zip(labels, axes, strict=True):
        for arm in ARMS:
            contents = np.asarray(spectra[key, arm]).cumsum(axis=1)
            lo, median, hi = np.quantile(contents, [0.10, 0.50, 0.90], axis=0)
            ranks = np.arange(1, len(median) + 1)
            ax.semilogx(ranks, median, label=LABELS[arm], color=COLORS[arm])
            ax.fill_between(ranks, lo, hi, color=COLORS[arm], alpha=0.12)
        ax.set(title=title, xlabel="Eigenvalue rank", ylim=(0, 1.025))
        ax.grid(axis="y", alpha=0.21)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Cumulative trace share")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_agreement(path, rows):
    fig, axes = plt.subplots(2, 2, figsize=(8.7, 5.5), layout="constrained", sharex=True)
    palette = dict(zip(combinations(ARMS, 2), ("#197555", "#8152a4", "#a36038")))
    for i, scale in enumerate(SCALES):
        for j, (metric, title) in enumerate((
            ("rank_spearman", "First-step rank correlation"), ("batch_overlap", "Batch overlap"),
        )):
            ax = axes[i, j]
            for (left, right), color in palette.items():
                selected = [row for row in rows if (row["comparison"], row["scale"], row["left"], row["right"])
                            == ("across_kernels", scale, left, right)]
                noises = [row["hypothetical_noise"] for row in selected]
                values = [row[metric] for row in selected]
                ax.semilogx(noises, [v["mean"] for v in values], "o-", color=color,
                            label=f"{LABELS[left]} vs {LABELS[right]}", markersize=3)
                if values[0]["ci95"] is not None:
                    low, high = np.array([v["ci95"] for v in values]).T
                    ax.fill_between(noises, low, high, color=color, alpha=0.13)
            ax.axvline(OBSERVED_NOISE, ls="--", color="#777777", lw=0.9)
            ax.set(ylim=(-0.03, 1.045))
            ax.grid(axis="y", alpha=0.21)
            ax.spines[["right", "top"]].set_visible(False)
            if j:
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            else:
                ax.set_ylabel(SCALE_LABELS[scale])
            if i == 0:
                ax.set_title(title)
            else:
                ax.set_xlabel("Hypothetical noise")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory", default="transport", choices=ARMS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    spectra = defaultdict(list)
    noise_rows, structure_rows, comparison_rows, pair_rows = [], [], [], []
    for checkpoint in replay_grams(args.checkpoint_dir, args.trajectory):
        parts = analyze_checkpoint(*checkpoint, spectra)
        for target, added in zip((noise_rows, structure_rows, comparison_rows, pair_rows), parts, strict=True):
            target.extend(added)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "noise_sweep": noise_rows, "conditioning_spectra": structure_rows,
        "ranking_comparisons": comparison_rows, "kernel_pairs": pair_rows,
    }
    for name, rows in tables.items():
        write_csv(args.output_dir / f"{name}.csv", rows)
    noise = summarize_groups(noise_rows, ("scale", "arm", "hypothetical_noise"), [
        k for k in noise_rows[0] if "_vs_noise_" in k
        or k.startswith(("share_candidate_", "first_score_", "hypothetical_noise_over_"))
    ])
    summary = {
        "trajectory": args.trajectory, "seeds": sorted({r["seed"] for r in noise_rows}),
        "checkpoint_rounds": sorted({r["round"] for r in noise_rows}),
        "observed_noise": OBSERVED_NOISE, "hypothetical_noise_grid": NOISES,
        "reference_hypothetical_noise": HISTORICAL_NOISE,
        "pool_score": "sum over the full pool of squared posterior column entries, divided by candidate variance plus hypothetical noise",
        "spectral_metrics": "eigenvalue masses are trace normalized and tiny negative numerical eigenvalues contribute zero mass",
        "effective_rank": "inverse sum of squared trace-normalized eigenvalue masses",
        "uncertainty": "10,000 percentile bootstrap resamples of paired seeds after averaging checkpoints within seed",
        "noise_sweep": noise,
        "condition_and_spectra": summarize_groups(structure_rows, ("scale", "arm"), [
            k for k in structure_rows[0] if k not in ("seed", "round", "trajectory", "scale", "arm")
        ]),
        "ranking_comparisons": summarize_groups(comparison_rows, ("comparison", "scale", "left", "right",
                                                                  "hypothetical_noise"), [
            "rank_spearman", "top10_overlap", "first_match", "batch_overlap", "exact_ordered_batch_match"
        ]),
        "kernel_pairs": summarize_groups(pair_rows, ("left", "right"), [
            k for k in pair_rows[0] if k not in ("seed", "round", "trajectory", "left", "right")
        ]),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot_sweep(args.output_dir / "noise_sensitivity.png", noise)
    plot_spectra(args.output_dir / "gram_spectra.png", spectra)
    plot_agreement(args.output_dir / "kernel_agreement.png", summary["ranking_comparisons"])


if __name__ == "__main__":
    main()
