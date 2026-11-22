"""Summarize the paired GP ablations and visualize actual acquisition histories."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


PROTOCOLS = {
    "original": "Original",
    "normalized": "Unit mean diagonal",
    "coherent": "Coherent noise",
    "normalized_coherent": "Both",
}
ARMS = {
    "transport": ("tNTK", "#245c91"),
    "projected": ("Projected pullback", "#197555"),
    "projected_canonical": ("Projected canonical", "#8152a4"),
    "mean": ("Mean NTK", "#ad4b37"),
    "mean_with_bias": ("Mean NTK + bias", "#9a5d22"),
    "random": ("Random", "#62676b"),
}
GEOMETRY_ARMS = ("transport", "projected", "projected_canonical")
COMPONENTS = ("total", "mean", "covariance")
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 2026


def load_metrics(path, metadata):
    rows = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            policy = row["protocol"], row["arm"]
            key = int(row["seed"]), int(row["round"])
            if key in rows[policy]:
                raise ValueError(f"Duplicate policy, seed and round {policy}, {key}")
            rows[policy][key] = (
                int(row["budget"]), [float(row[name]) for name in COMPONENTS],
                json.loads(row["acquired_indices"]), json.loads(row["selected_x1"]),
                json.loads(row["selected_x2"]),
            )

    seeds = metadata["seeds"]
    config = metadata["config"]
    rounds = range(config["n_rounds"] + 1)
    budgets = [config["n_init"] + r * config["batch_size"] for r in rounds]
    expected = {(seed, rnd) for seed in seeds for rnd in rounds}
    if set(rows) != {tuple(item) for item in metadata["policies"]}:
        raise ValueError("Policies do not match the run manifest")
    curves, batches, locations = {}, {}, {}
    reference = rows["original", "transport"]
    for policy, history in rows.items():
        if set(history) != expected:
            raise ValueError(f"Incomplete paired trajectory for {policy}")
        curves[policy] = np.asarray([[history[seed, rnd][1] for rnd in rounds] for seed in seeds])
        if not np.isfinite(curves[policy]).all():
            raise ValueError(f"Nonfinite losses for {policy}")
        if not np.allclose(curves[policy][:, :, 0], curves[policy][:, :, 1:].sum(axis=-1), atol=2e-7):
            raise ValueError(f"Components do not sum to the total for {policy}")
        batches[policy], locations[policy] = [], []
        for seed in seeds:
            if history[seed, 0] != reference[seed, 0]:
                raise ValueError(f"Unpaired starting point for {policy}, seed {seed}")
            known, seed_batches, seed_locations = set(), [], []
            for rnd, budget in enumerate(budgets):
                recorded_budget, _, ids, x1, x2 = history[seed, rnd]
                if (recorded_budget != budget or len(ids) != budget - len(known)
                        or len(set(ids)) != len(ids) or known.intersection(ids)):
                    raise ValueError(f"Invalid acquisition history for {policy}, seed {seed}, round {rnd}")
                known.update(ids)
                if rnd:
                    seed_batches.append(set(ids))
                    seed_locations.extend(zip(x1, x2, strict=True))
            batches[policy].append(seed_batches)
            locations[policy].append(seed_locations)
        locations[policy] = np.asarray(locations[policy])
    return budgets, curves, batches, locations


def statistics(values, draw_indices):
    interval = None
    se = None
    if len(values) > 1:
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
        interval = np.quantile(values[draw_indices].mean(axis=1), [0.025, 0.975]).tolist()
    return {"mean": float(values.mean()), "se": se, "ci95": interval}


def acquisitions(batches, left, right):
    for first, second in zip(batches[left], batches[right], strict=True):
        step = [len(a & b) / len(a) for a, b in zip(first, second, strict=True)]
        a, b = set.union(*first), set.union(*second)
        yield [*step, len(a & b) / len(a)]


def summarize(metadata, budgets, curves, batches, locations, output):
    seeds = metadata["seeds"]
    draw_indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        len(seeds), size=(N_BOOTSTRAP, len(seeds)),
    )
    endpoints = {}
    for policy, curve in curves.items():
        endpoints[policy] = {
            "final": curve[:, -1, :],
            "post_round_mean": curve[:, 1:, :].mean(axis=1),
            "post_round_mean_minus_initial": curve[:, 1:, :].mean(axis=1) - curve[:, 0, :],
        }

    def compare(left, right=None):
        return {
            endpoint: {
                component: statistics(
                    values[:, i] if right is None else values[:, i] - endpoints[right][endpoint][:, i],
                    draw_indices,
                )
                for i, component in enumerate(COMPONENTS)
            }
            for endpoint, values in endpoints[left].items()
        }

    contrasts = {}
    with (output / "paired_acquisitions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("contrast", "seed", *(f"round_{r}_overlap" for r in range(1, len(budgets))),
                         "final_acquired_set_overlap"))

        def overlap(label, left, right):
            values = np.asarray(list(acquisitions(batches, left, right)))
            for seed, row in zip(seeds, values, strict=True):
                writer.writerow((label, seed, *row))
            return {
                "by_round": [statistics(values[:, i], draw_indices) for i in range(values.shape[1] - 1)],
                "final_acquired_set": statistics(values[:, -1], draw_indices),
            }

        shared_random = ("shared", "random")
        for protocol in PROTOCOLS:
            reference = protocol, "transport"
            arms = [policy for policy in curves if policy[0] == protocol] + [shared_random]
            contrasts[protocol] = {}
            for policy in arms:
                arm = policy[1]
                x1 = locations[policy][:, :, 0]
                arm_summary = {
                    "metrics": compare(policy),
                    "x1_region_fractions": {
                        "left": statistics((x1 < 1 / 3).mean(axis=1), draw_indices),
                        "middle": statistics(((x1 >= 1 / 3) & (x1 < 2 / 3)).mean(axis=1), draw_indices),
                        "right": statistics((x1 >= 2 / 3).mean(axis=1), draw_indices),
                    },
                    "kernel_config": metadata["protocols"][protocol] if arm != "random" else None,
                }
                if policy != reference:
                    arm_summary["difference_vs_transport"] = compare(policy, reference)
                    arm_summary["overlap_vs_transport"] = overlap(
                        f"{protocol}:{arm}_vs_transport", policy, reference,
                    )
                if protocol != "original" and arm != "random":
                    arm_summary["overlap_vs_same_arm_original"] = overlap(
                        f"{protocol}:{arm}_vs_original", policy, ("original", arm),
                    )
                contrasts[protocol][arm] = arm_summary

    sensitivity = {}
    for arm in GEOMETRY_ARMS[1:]:
        gain = {protocol: endpoints[protocol, arm]["final"][:, 0]
                - endpoints[protocol, "transport"]["final"][:, 0] for protocol in PROTOCOLS}
        sensitivity[arm] = {
            "change_from_original": {
                protocol: statistics(gain[protocol] - gain["original"], draw_indices)
                for protocol in PROTOCOLS if protocol != "original"
            },
            "normalization_at_coherent_noise": statistics(gain["normalized_coherent"] - gain["coherent"], draw_indices),
            "coherence_at_unit_diagonal": statistics(gain["normalized_coherent"] - gain["normalized"], draw_indices),
        }
    return {
        "primary": "Final total W2 squared, projected minus transport under the original protocol",
        "bootstrap": {"draws": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "unit": "paired seed",
                      "intervals": "two-sided percentile, unadjusted"},
        "seeds": seeds,
        "budgets": budgets,
        "protocols": contrasts,
        "gain_sensitivity": sensitivity,
        "mean_bias_contrast": compare(("original", "mean_with_bias"), ("original", "mean")),
    }


def plot_curves(path, budgets, curves):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained", sharex=True, sharey=True)
    legend = {}
    for (protocol, title), ax in zip(PROTOCOLS.items(), axes.flat, strict=True):
        arms = [policy for policy in curves if policy[0] == protocol] + [("shared", "random")]
        for policy in arms:
            values = curves[policy][:, :, 0]
            label, color = ARMS[policy[1]]
            mean = values.mean(axis=0)
            line, = ax.plot(budgets, mean, ".-", color=color)
            legend[label] = line
            if len(values) > 1:
                se = values.std(axis=0, ddof=1) / np.sqrt(len(values))
                ax.fill_between(budgets, mean - se, mean + se, color=color, alpha=0.10)
        ax.set(title=title, xticks=budgets)
        ax.grid(axis="y", alpha=0.22)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Test $W_2^2$")
    for ax in axes[-1]:
        ax.set_xlabel("Labeled conditions")
    fig.legend(legend.values(), legend.keys(), loc="outside upper center", frameon=False, ncol=3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_contrasts(path, summary):
    fig, (final, trajectory) = plt.subplots(1, 2, figsize=(10, 4.7), sharex=True, sharey=True,
                                           layout="constrained")
    for ax, endpoint, title in ((final, "final", "Final test loss"),
                                (trajectory, "post_round_mean", "Mean test loss across rounds 1–5")):
        for i, (protocol, label) in enumerate(PROTOCOLS.items()):
            for arm, offset in (("projected", -0.13), ("projected_canonical", 0.13)):
                stats = summary["protocols"][protocol][arm]["difference_vs_transport"][endpoint]["total"]
                center = stats["mean"]
                interval = stats["ci95"]
                ax.plot(center, i + offset, "o", color=ARMS[arm][1], label=ARMS[arm][0] if i == 0 else None)
                if interval is not None:
                    ax.hlines(i + offset, *interval, color=ARMS[arm][1], linewidth=1.5)
        ax.axvline(0, color="#6f757b", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel=r"$W_2^2$ difference, projected minus tNTK")
        ax.set_yticks(range(len(PROTOCOLS)), PROTOCOLS.values())
        ax.grid(axis="x", alpha=0.2)
    final.invert_yaxis()
    fig.legend(*final.get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_regions(path, locations, n_side):
    fig, axes = plt.subplots(len(PROTOCOLS), len(GEOMETRY_ARMS), figsize=(8.6, 8.9),
                             sharex=True, sharey=True, layout="constrained")
    n_seeds = len(locations["original", "transport"])
    grid = np.linspace(0, 1, n_side)
    half = 0.5 / (n_side - 1)
    for row, (protocol, label) in enumerate(PROTOCOLS.items()):
        for col, arm in enumerate(GEOMETRY_ARMS):
            ax = axes[row, col]
            loc = locations[protocol, arm].reshape(-1, 2)
            edge = np.r_[-half, (grid[:-1] + grid[1:]) / 2, 1 + half]
            frequency, _, _ = np.histogram2d(loc[:, 0], loc[:, 1], bins=(edge, edge))
            image = ax.imshow(frequency.T / n_seeds, origin="lower", cmap="Blues", vmin=0, vmax=1,
                              extent=(-half, 1 + half, -half, 1 + half))
            ax.set(xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
            if row == 0:
                ax.set_title(ARMS[arm][0])
            if col == 0:
                ax.set_ylabel(label + "\n" + r"$x_2$")
            if row == len(PROTOCOLS) - 1:
                ax.set_xlabel(r"$x_1$")
    fig.colorbar(image, ax=axes, shrink=0.55, label="Fraction of seeds acquiring the condition")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    budgets, curves, batches, locations = load_metrics(args.input_dir / "active_learning.csv", manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(manifest, budgets, curves, batches, locations, args.output_dir)
    (args.output_dir / "gp_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_curves(args.output_dir / "learning_curves.png", budgets, curves)
    plot_contrasts(args.output_dir / "paired_contrasts.png", summary)
    plot_regions(args.output_dir / "spatial_acquisitions.png", locations,
                 int(np.round(np.sqrt(manifest["config"]["n_pool"]))))


if __name__ == "__main__":
    main()
