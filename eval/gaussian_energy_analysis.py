"""Analyze tangent energy and counterfactual acquisition on Gaussian checkpoints."""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, PercentFormatter


REFERENCE = "transport"
VARIANTS = ("projected", "projected_canonical")
VARIANT_LABELS = ("Same-latent projection", "Canonical projection")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2026
ENERGY_THRESHOLD = 0.10


def read_checkpoints(path):
    """Group point measurements by seed, fitted policy and round."""
    points = defaultdict(list)
    fields = (
        "pool_index", "x1", "x2", "is_labeled", "diag_transport", "diag_projected",
        "diag_transport_cov", "diag_projected_cov",
        "select_transport", "select_projected", "select_projected_canonical",
    )
    optional = tuple(f"select_{arm}_matched_diag" for arm in VARIANTS)
    open_text = gzip.open if path.suffix == ".gz" else open
    with open_text(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"checkpoint_policy", "seed", "round", *fields}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing checkpoint columns: {', '.join(sorted(missing))}")
        fields += tuple(field for field in optional if field in reader.fieldnames)
        for row in reader:
            key = (row["checkpoint_policy"], int(row["seed"]), int(row["round"]))
            points[key].append([float(row[field]) for field in fields])
    if not points:
        raise ValueError("No saved checkpoints")
    histories, seeds, rounds = map(set, zip(*points))
    if rounds != set(range(max(rounds) + 1)) or len(points) != len(histories) * len(seeds) * len(rounds):
        raise ValueError("Expected every policy and round for the same set of seeds")
    for key, rows in points.items():
        data = np.asarray(rows)
        points[key] = dict(zip(fields, data.T))
    return points


def residualize_axes(values, x1, x2):
    """Remove additive grid-coordinate effects within one eligible pool."""
    _, first = np.unique(x1, return_inverse=True)
    _, second = np.unique(x2, return_inverse=True)
    # The x2 baseline is absorbed by the complete set of x1 indicators.
    design = np.column_stack((np.eye(first.max() + 1)[first], np.eye(second.max() + 1)[second, 1:]))
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def batch_overlap(first, second):
    return float(np.count_nonzero(first & second) / np.count_nonzero(first))


def measure_checkpoints(checkpoints):
    records, spatial = [], defaultdict(list)
    reference_grid = None
    for (history, seed, round_index), data in sorted(checkpoints.items()):
        order = np.argsort(data["pool_index"])
        data = {name: values[order] for name, values in data.items()}
        pool_index, x1, x2 = (data[field] for field in ("pool_index", "x1", "x2"))
        grid = np.column_stack((pool_index, x1, x2))
        if reference_grid is None:
            reference_grid = grid
        if not np.array_equal(grid, reference_grid) or len(np.unique(pool_index)) != len(pool_index):
            raise ValueError("The benchmark pool must be fixed and contain unique indices")
        raw, projected, raw_cov, projected_cov = (
            data[field] for field in (
                "diag_transport", "diag_projected", "diag_transport_cov", "diag_projected_cov"
            )
        )
        location = (history, seed, round_index)
        if not np.isfinite(grid).all() or not np.isfinite([raw, projected, raw_cov, projected_cov]).all():
            raise ValueError(f"Nonfinite grid or diagonal at {location}")
        if np.any(raw <= 0) or np.any(raw_cov <= 0):
            raise ValueError(f"Raw tangent-energy denominator must be positive at {location}")
        energy = 1 - projected / raw
        cov_energy = 1 - projected_cov / raw_cov
        # Check contraction and the unchanged mean block, allowing float32 roundoff.
        if (np.any((energy < -2e-5) | (energy > 1 + 2e-5))
            or np.any((cov_energy < -2e-5) | (cov_energy > 1 + 2e-5))
            or not np.allclose(raw - projected, raw_cov - projected_cov, rtol=2e-5, atol=2e-5)):
            raise ValueError(f"Inconsistent projected energy or mean term at {location}")

        indicator_columns = {field: values for field, values in data.items()
                             if field == "is_labeled" or field.startswith("select_")}
        if any(not np.isin(values, [0, 1]).all() for values in indicator_columns.values()):
            raise ValueError(f"Expected binary labels and selections at {location}")
        eligible = ~indicator_columns.pop("is_labeled").astype(bool)
        batches = {field.removeprefix("select_"): values.astype(bool)
                   for field, values in indicator_columns.items()}
        reference = batches[REFERENCE]
        if not np.any(reference) or any(
            np.count_nonzero(batch) != np.count_nonzero(reference) or np.any(batch & ~eligible)
            for batch in batches.values()
        ):
            raise ValueError(f"Inconsistent counterfactual batch at {location}")

        adjusted = np.full_like(energy, np.nan)
        adjusted[eligible] = residualize_axes(energy[eligible], x1[eligible], x2[eligible])
        totals = {}
        for label, values in (("total", energy), ("covariance", cov_energy)):
            for prefix, pool in ((label, values[eligible]), (f"full_pool_{label}", values)):
                p10, p50, p90 = np.quantile(pool, [0.10, 0.50, 0.90])
                totals.update({
                    f"{prefix}_mean": float(pool.mean()), f"{prefix}_p10": float(p10),
                    f"{prefix}_p50": float(p50), f"{prefix}_p90": float(p90),
                    f"{prefix}_share_ge_10pct": float(np.mean(pool >= ENERGY_THRESHOLD)),
                })
        totals["total_trace_weighted"] = float(np.sum((raw - projected)[eligible]) / raw[eligible].sum())
        totals["covariance_trace_weighted"] = float(
            np.sum((raw_cov - projected_cov)[eligible]) / raw_cov[eligible].sum()
        )
        totals["full_pool_total_trace_weighted"] = float(np.sum(raw - projected) / raw.sum())
        totals["full_pool_covariance_trace_weighted"] = float(
            np.sum(raw_cov - projected_cov) / raw_cov.sum()
        )
        projected_pair_overlap = batch_overlap(*(batches[variant] for variant in VARIANTS))
        for variant in VARIANTS:
            projected_batch = batches[variant]
            reference_only = reference & ~projected_batch
            projected_only = projected_batch & ~reference
            record = dict(
                checkpoint_policy=history, seed=seed, round=round_index, variant=variant,
                **totals, n_eligible=int(eligible.sum()), batch_size=int(reference.sum()),
                n_exclusive_per_batch=int(reference_only.sum()),
                overlap=batch_overlap(reference, projected_batch),
                projected_pair_overlap=projected_pair_overlap,
                selected_energy_difference=float(energy[reference].mean() - energy[projected_batch].mean()),
                selected_cov_energy_difference=float(
                    cov_energy[reference].mean() - cov_energy[projected_batch].mean()
                ),
                exclusive_energy_difference=None, exclusive_adjusted_energy_difference=None,
                exclusive_x1_difference=None, exclusive_x2_difference=None,
            )
            if np.any(reference_only):
                for key, values in (
                    ("exclusive_energy_difference", energy),
                    ("exclusive_adjusted_energy_difference", adjusted),
                    ("exclusive_x1_difference", x1),
                    ("exclusive_x2_difference", x2),
                ):
                    record[key] = float(values[reference_only].mean() - values[projected_only].mean())
            matched = batches.get(f"{variant}_matched_diag")
            if matched is not None:
                record["matched_diag_overlap"] = batch_overlap(reference, matched)
                record["matched_diag_change_in_overlap"] = record["matched_diag_overlap"] - record["overlap"]
                record["projected_vs_matched_diag_overlap"] = batch_overlap(projected_batch, matched)
            records.append(record)
            spatial[(history, variant)].append(np.stack((
                eligible, np.where(eligible, energy, 0), reference, projected_batch
            )))
    return records, reference_grid, spatial


def paired_bootstrap(records, metric):
    """Average rounds within seed and resample seeds as paired units."""
    grouped = defaultdict(list)
    for row in records:
        value = row[metric]
        if value is not None:
            grouped[row["seed"]].append(value)
    means = np.asarray([np.mean(values) for _, values in sorted(grouped.items())])
    if not len(means):
        return dict(mean=None, ci95=None, n_seeds=0, n_checkpoints=0)
    ci95 = None
    if len(means) > 1:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        sampled = rng.choice(means, size=(BOOTSTRAP_DRAWS, len(means)), replace=True)
        ci95 = np.quantile(sampled.mean(axis=1), [0.025, 0.975]).tolist()
    return dict(mean=float(means.mean()), ci95=ci95, n_seeds=len(means),
                n_checkpoints=sum(map(len, grouped.values())))


def summarize(records):
    histories = sorted({row["checkpoint_policy"] for row in records})
    by_history = {}
    energy_metrics = [field for field in records[0]
                      if field.startswith(("total_", "covariance_", "full_pool_"))]
    acquisition_metrics = [
        "overlap", "selected_energy_difference", "selected_cov_energy_difference",
        "exclusive_energy_difference", "exclusive_adjusted_energy_difference",
        "exclusive_x1_difference", "exclusive_x2_difference",
    ]
    acquisition_metrics += [field for field in records[0] if "matched_diag" in field]
    for history in histories:
        subset = [row for row in records if row["checkpoint_policy"] == history]
        rounds = sorted({row["round"] for row in subset})
        # Final hypothetical batches are exported separately from campaign rounds.
        actual_rounds = rounds[:-1]
        current = {}
        pool = [row for row in subset if row["variant"] == VARIANTS[0]]
        current["all_rounds_pool_energy"] = {
            metric: paired_bootstrap(pool, metric) for metric in energy_metrics
        }
        current["pool_energy_by_round"] = {
            round_index: {metric: paired_bootstrap(
                [row for row in pool if row["round"] == round_index], metric
            ) for metric in energy_metrics} for round_index in rounds
        }
        initial = {row["seed"]: row for row in pool if row["round"] == rounds[0]}
        final = {row["seed"]: row for row in pool if row["round"] == rounds[-1]}
        current["pool_energy_final_minus_initial"] = {
            metric: paired_bootstrap([
                {"seed": seed, metric: row[metric] - initial[seed][metric]} for seed, row in final.items()
            ], metric) for metric in energy_metrics
        }
        current["between_projected_acquisition"] = {
            "campaign_rounds_overlap": paired_bootstrap(
                [row for row in pool if row["round"] in actual_rounds], "projected_pair_overlap"
            ),
            "by_round_overlap": {
                round_index: paired_bootstrap(
                    [row for row in pool if row["round"] == round_index], "projected_pair_overlap"
                ) for round_index in rounds
            },
        }
        selections = {}
        for variant in VARIANTS:
            comparisons = [row for row in subset if row["variant"] == variant]
            selections[variant] = {
                "campaign_rounds": {metric: paired_bootstrap(
                    [row for row in comparisons if row["round"] in actual_rounds], metric
                ) for metric in acquisition_metrics},
                "by_round": {round_index: {metric: paired_bootstrap(
                    [row for row in comparisons if row["round"] == round_index], metric
                ) for metric in acquisition_metrics} for round_index in rounds},
            }
        current["same_checkpoint_acquisition"] = selections
        current["campaign_acquisition_rounds"] = actual_rounds
        current["hypothetical_final_acquisition_round"] = rounds[-1]
        by_history[history] = current
    return {
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "seed"},
        "default_pool": "unlabeled candidates",
        "full_pool_prefix": "metrics prefixed full_pool include every point, including labeled conditions",
        "pool_quantiles": "compute quantiles in each checkpoint pool, then average over seeds",
        "energy": "1 - projected diagonal / raw diagonal",
        "trace_weighted_energy": "sum(raw diagonal - projected diagonal) / sum(raw diagonal)",
        "enrichment_sign": "transport minus projected, with positive values indicating higher removed energy under transport selection",
        "exclusive_contrast": "compare disjoint parts of equal-sized batches on checkpoints with disagreement",
        "adjustment": "residualize r on additive x1 and x2 fixed effects within the eligible pool at each checkpoint",
        "matched_diag": "rescale projected Gram by congruence to match the raw diagonal while preserving projected correlations",
        "by_history": by_history,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def spatial_rows(grid, spatial, records):
    rows = []
    round_lists = defaultdict(list)
    for row in records:
        round_lists[(row["checkpoint_policy"], row["variant"])].append(row["round"])
    for (history, variant), values in sorted(spatial.items()):
        rounds = round_lists[(history, variant)]
        selected_rounds = np.asarray(rounds) < max(rounds)
        summed = np.sum(np.asarray(values)[selected_rounds], axis=0)
        exposures, energy_sum, total_raw, total_proj = summed
        for (pool_index, x1, x2), n, energy, raw, projected in zip(
            grid, exposures, energy_sum, total_raw, total_proj
        ):
            rows.append(dict(
                checkpoint_policy=history, variant=variant, pool_index=int(pool_index),
                x1=float(x1), x2=float(x2), eligible_checkpoints=int(n),
                mean_energy=float(energy / n) if n else None,
                n_selected_transport=int(raw), n_selected_projected=int(projected),
                acquisition_rate_difference=float((raw - projected) / n) if n else None,
            ))
    return rows


def plot_energy_dynamics(path, history):
    energy = history["pool_energy_by_round"]
    rounds = list(energy)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.35), layout="constrained", sharex=True)
    for axis, block, label, color in zip(
        axes, ("total", "covariance"), ("All head parameters", "Covariance head"),
        ("#245984", "#2c7d67"),
    ):
        quantiles = np.array([[energy[r][f"full_pool_{block}_p{q}"]["mean"] for r in rounds]
                              for q in (10, 50, 90)])
        eligible = [energy[r][f"{block}_p50"]["mean"] for r in rounds]
        axis.fill_between(rounds, quantiles[0], quantiles[2], color=color, alpha=0.18,
                          label="Full pool P10–P90")
        axis.plot(rounds, quantiles[1], "o-", color=color, markersize=4, label="Full pool median")
        axis.plot(rounds, eligible, "--", color="#444444", linewidth=1.0, label="Unlabeled median")
        axis.set(title=label, xlabel="Acquisition round", xticks=rounds)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=0.28)
        axis.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Fraction of diagonal energy removed")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_spatial_maps(path, spatial, history):
    rows = [[r for r in spatial if r["checkpoint_policy"] == history and r["variant"] == variant]
            for variant in VARIANTS]
    coords = np.array([[r["x1"], r["x2"]] for r in rows[0]])
    energy = [np.nan if r["mean_energy"] is None else r["mean_energy"] for r in rows[0]]
    differences = [[np.nan if r["acquisition_rate_difference"] is None else r["acquisition_rate_difference"]
                   for r in variant] for variant in rows]
    max_abs = max(float(np.nanmax(np.abs(differences))), 1e-5)
    fig, axes = plt.subplots(1, 3, figsize=(10.75, 3.6), layout="constrained", sharey=True)
    energy_plot = axes[0].scatter(*coords.T, c=energy, cmap="viridis", marker="s", s=30)
    for axis, difference, label in zip(axes[1:], differences, VARIANT_LABELS):
        diff_plot = axis.scatter(*coords.T, c=difference, cmap="RdBu_r",
                                 norm=TwoSlopeNorm(0, -max_abs, max_abs), marker="s", s=30)
        axis.set_title(label)
    axes[0].set(title="Removed energy", ylabel="$x_2$")
    for axis in axes:
        axis.set(xlabel="$x_1$", xlim=(-0.06, 1.06), ylim=(-0.06, 1.06), aspect="equal")
    fig.colorbar(energy_plot, ax=axes[0], format=PercentFormatter(1), shrink=0.77, pad=0.03)
    fig.colorbar(diff_plot, ax=list(axes[1:]), format=PercentFormatter(1), shrink=0.77, pad=0.03,
                 label="Selection rate difference (tNTK − projected)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_acquisition_contrast(path, history):
    selectors = history["same_checkpoint_acquisition"]
    metrics = ("selected_energy_difference", "exclusive_energy_difference", "exclusive_adjusted_energy_difference")
    labels = ("Whole batches", "Exclusive choices", "Exclusive, spatially adjusted")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.25), layout="constrained")
    for row, (variant, label, color) in enumerate(zip(VARIANTS, VARIANT_LABELS, ("#246f5a", "#784d92"))):
        stats = selectors[variant]["campaign_rounds"]
        for i, metric in enumerate(metrics):
            value = stats[metric]
            if value["mean"] is None:
                continue
            y = i + (row - 0.5) * 0.16
            ci = value["ci95"]
            axes[0].plot(value["mean"], y, "o", color=color, label=label if i == 0 else None)
            if ci:
                axes[0].hlines(y, *ci, color=color, linewidth=1.55)
        overlap = ("overlap", "matched_diag_overlap")
        for i, metric in enumerate(overlap):
            if metric not in stats:
                continue
            y = i + (row - 0.5) * 0.16
            ci = stats[metric]["ci95"]
            axes[1].plot(stats[metric]["mean"], y, "o", color=color)
            if ci:
                axes[1].hlines(y, *ci, color=color, linewidth=1.55)
    axes[0].axvline(0, color="#767676", lw=0.85, zorder=0)
    axes[0].set(xlabel="Removed energy, tNTK − projected (percentage points)",
                yticks=range(len(labels)), yticklabels=labels, ylim=(2.45, -0.45))
    axes[1].set(xlabel="Batch overlap with tNTK", yticks=[0, 1],
                yticklabels=["Original diagonals", "Matched diagonals"], ylim=(1.45, -0.45), xlim=(-0.04, 1.04))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False)
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{100 * x:g}"))
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
        axis.spines[["right", "top"]].set_visible(False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Per-condition CSV or CSV.gz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-history", default=REFERENCE, help="Fitted policy used in figures")
    args = parser.parse_args()

    records, grid, pool = measure_checkpoints(read_checkpoints(args.input))
    summary = summarize(records)
    if args.plot_history not in summary["by_history"]:
        parser.error(f"Unknown --plot-history {args.plot_history!r}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spatial = spatial_rows(grid, pool, records)
    history = summary["by_history"][args.plot_history]
    write_csv(args.output_dir / "checkpoint_metrics.csv", records)
    write_csv(args.output_dir / "spatial_metrics.csv", spatial)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_energy_dynamics(args.output_dir / "energy_dynamics.png", history)
    plot_spatial_maps(args.output_dir / "energy_acquisitions.png", spatial, args.plot_history)
    plot_acquisition_contrast(args.output_dir / "energy_contrasts.png", history)


if __name__ == "__main__":
    main()
