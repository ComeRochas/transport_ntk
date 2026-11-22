"""Summarize a paired 2D Gaussian active-learning run saved by the trainer."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt


METRICS = ("total", "mean", "covariance")
REFERENCE = "transport"
BOOTSTRAP_SEED = 2026
BOOTSTRAP_DRAWS = 10_000
ARM_LABELS = {
    "mean": ("Mean NTK", "#b04a34"),
    "transport": ("Transport NTK", "#2c5d91"),
    "random": ("Random", "#6f757b"),
    "projected": ("Projected NTK", "#22785b"),
    "projected_canonical": ("Canonical projected NTK", "#7b5396"),
}


def load_results(path):
    config = json.loads(path.with_name("config.json").read_text())
    data = torch.load(path, map_location="cpu", weights_only=True)
    results, indices = data["results"], data["indices"]
    if REFERENCE not in results or results.keys() != indices.keys():
        raise ValueError("Results must include the transport arm and acquisitions for every arm")

    rounds = range(config["n_rounds"] + 1)
    budgets = [config["n_init"] + r * config["batch_size"] for r in rounds]
    n_seeds = len(results[REFERENCE])
    if n_seeds == 0 or len(budgets) < 2:
        raise ValueError("At least one seed and one acquisition round are required")

    curves = {}
    for arm, runs in results.items():
        if len(runs) != n_seeds or len(indices[arm]) != n_seeds:
            raise ValueError(f"Different numbers of paired seeds for {arm}")
        for seed, (run, selected) in enumerate(zip(runs, indices[arm])):
            if set(run) != set(rounds):
                raise ValueError(f"Unexpected rounds for {arm}, seed {seed}")
            if len(selected) != budgets[-1] or len(set(selected)) != len(selected):
                raise ValueError(f"Unexpected acquisition count or duplicates for {arm}, seed {seed}")
            if selected[:budgets[0]] != indices[REFERENCE][seed][:budgets[0]]:
                raise ValueError(f"Initial acquisitions differ for {arm}, seed {seed}")
        curves[arm] = np.array([[[run[r][metric] for metric in METRICS] for r in rounds]
                                 for run in runs], dtype=float)
        if not np.isfinite(curves[arm]).all():
            raise ValueError(f"Nonfinite metric for {arm}")

    initial = curves[REFERENCE][:, 0, :]
    for arm, curve in curves.items():
        if not np.allclose(curve[:, 0, :], initial, rtol=1e-6, atol=1e-7):
            raise ValueError(f"Initial metrics differ between {arm} and {REFERENCE}")
    return config, budgets, curves, indices


def write_metrics(path, budgets, curves, indices):
    columns = ("seed", "round", "arm", "budget", "w2_sq_total", "w2_sq_mean",
               "w2_sq_covariance", "acquired_indices")
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(columns)
        for arm, curve in curves.items():
            for seed, (run, selected) in enumerate(zip(curve, indices[arm])):
                start = 0
                for round_idx, budget in enumerate(budgets):
                    acquired = json.dumps(selected[start:budget], separators=(",", ":"))
                    writer.writerow((seed, round_idx, arm, budget, *run[round_idx], acquired))
                    start = budget


def summary_statistics(values, bootstrap_indices=None):
    stats = {"mean": float(np.mean(values)), "se": None}
    if len(values) > 1:
        stats["se"] = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    if bootstrap_indices is not None:
        stats["ci95"] = None
        if len(values) > 1:
            draws = values[bootstrap_indices].mean(axis=1)
            stats["ci95"] = np.quantile(draws, [0.025, 0.975]).tolist()
    return stats


def make_summary(config, budgets, curves):
    n_seeds = len(curves[REFERENCE])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(n_seeds, size=(BOOTSTRAP_DRAWS, n_seeds))
    endpoints = {
        arm: {"final": curve[:, -1, :],
              "post_round_average_minus_initial": curve[:, 1:, :].mean(axis=1) - curve[:, 0, :]}
        for arm, curve in curves.items()
    }
    arms = {}
    for arm, arm_endpoints in endpoints.items():
        arms[arm] = {}
        for endpoint, values in arm_endpoints.items():
            arms[arm][endpoint] = {
                metric: summary_statistics(values[:, i]) for i, metric in enumerate(METRICS)
            }
        if arm != REFERENCE:
            arms[arm]["paired_difference_vs_transport"] = {
                endpoint: {
                    metric: summary_statistics(values[:, i] - endpoints[REFERENCE][endpoint][:, i],
                                               bootstrap_indices)
                    for i, metric in enumerate(METRICS)
                }
                for endpoint, values in arm_endpoints.items()
            }
    return {
        "config": config,
        "n_seeds": n_seeds,
        "budgets": budgets,
        "units": "squared Wasserstein distance (unscaled)",
        "primary_comparison": "projected minus transport, final total W2 squared",
        "bootstrap": {"method": "paired percentile bootstrap over seeds",
                      "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS},
        "arms": arms,
    }


def write_chart(path, budgets, curves):
    n_seeds = len(curves[REFERENCE])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.65), layout="constrained", sharex=True)
    panel_labels = ("Total", "Mean component", "Covariance component")
    for i, (axis, label) in enumerate(zip(axes, panel_labels)):
        for arm, curve in curves.items():
            name, color = ARM_LABELS[arm]
            values = curve[:, :, i]
            mean = values.mean(axis=0)
            axis.plot(budgets, mean, "o-", markersize=3.4, color=color, label=name)
            if n_seeds > 1:
                se = values.std(axis=0, ddof=1) / np.sqrt(n_seeds)
                axis.fill_between(budgets, mean - se, mean + se, color=color, alpha=0.12)
        axis.set(title=label, xlabel="Labeled conditions", xticks=budgets)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel(r"Test $W_2^2$")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center",
               ncol=len(curves), frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="results.pt from the trainer, next to config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config, budgets, curves, indices = load_results(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics(args.output_dir / "metrics.csv", budgets, curves, indices)
    summary = make_summary(config, budgets, curves)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_chart(args.output_dir / "learning_curves.png", budgets, curves)


if __name__ == "__main__":
    main()
