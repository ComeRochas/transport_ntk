"""Run the paired Gaussian selector ablations and record tangent-energy diagnostics."""

import argparse
import csv
import gzip
import json
import multiprocessing
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .experiment import (
    GaussianTransportNet,
    compute_mean_kernel,
    compute_transport_kernel,
    evaluate,
    get_distribution_parameters,
    load_config,
    make_grids,
    select_batch_min_total_var,
    train_model,
)
from .gaussian_projection import compute_projected_transport_kernel


PROTOCOLS = {
    "original": (False, False),
    "normalized": (True, False),
    "coherent": (False, True),
    "normalized_coherent": (True, True),
}
TRANSPORT_ARMS = ("transport", "projected", "projected_canonical")
KERNEL_ARMS = (*TRANSPORT_ARMS, "mean")
POLICIES = (
    *((protocol, arm) for protocol in PROTOCOLS for arm in KERNEL_ARMS),
    ("original", "mean_with_bias"),
    ("shared", "random"),
)
ENERGY_SELECTIONS = (
    *TRANSPORT_ARMS, "projected_matched_diag", "projected_canonical_matched_diag",
)
ENERGY_COLUMNS = (
    "seed", "round", "checkpoint_policy", "pool_index", "x1", "x2", "is_labeled",
    "diag_transport", "diag_projected", "diag_transport_cov", "diag_projected_cov",
    *(f"select_{arm}" for arm in ENERGY_SELECTIONS),
)
METRIC_COLUMNS = (
    "seed", "round", "protocol", "arm", "budget", "total", "mean", "covariance",
    "acquired_indices", "selected_x1", "selected_x2",
)


def gaussian_kernels(model, x):
    mean_weights = compute_mean_kernel(model, x)
    return {
        "transport": compute_transport_kernel(model, x),
        "projected": compute_projected_transport_kernel(model, x),
        "projected_canonical": compute_projected_transport_kernel(model, x, canonical=True),
        "mean": mean_weights,
        "mean_with_bias": mean_weights + 2.0,
    }


def select(K, indices, batch_size, lam, protocol):
    normalized, coherent = PROTOCOLS[protocol]
    if normalized:
        scale = K.diagonal().mean()
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError("Mean kernel diagonal must be finite and positive")
        K = K / scale
    return select_batch_min_total_var(
        K, range(len(K)), indices, batch_size, lam,
        hypothetical_lam=lam if coherent else 1.0,
    )


def energy_rows(seed, round_idx, policy, points, indices, kernels, config):
    diag = {arm: kernels[arm].diagonal() for arm in TRANSPORT_ARMS}
    torch.testing.assert_close(diag["projected"], diag["projected_canonical"])
    matched = {}
    for arm in TRANSPORT_ARMS[1:]:
        scales = (diag["transport"] / diag[arm]).sqrt()
        matched[f"{arm}_matched_diag"] = kernels[arm] * torch.outer(scales, scales)
    batches = {
        arm: set(select(K, indices, config.batch_size, config.lam, "original"))
        for arm, K in {**{arm: kernels[arm] for arm in TRANSPORT_ARMS}, **matched}.items()
    }
    mean_diag = kernels["mean_with_bias"].diagonal()
    diagnostics = zip(
        diag["transport"].tolist(), diag["projected"].tolist(),
        (diag["transport"] - mean_diag).tolist(), (diag["projected"] - mean_diag).tolist(),
    )
    known = set(indices)
    return [
        [seed, round_idx, policy, i, x1, x2, int(i in known), *values,
         *(int(i in batches[arm]) for arm in ENERGY_SELECTIONS)]
        for i, ((x1, x2), values) in enumerate(zip(points, diagnostics, strict=True))
    ]


def run_seed(seed, *, config):
    """Reuse fits only when the ordered labels and round RNG state are identical."""
    torch.manual_seed(seed)
    x_pool, x_test, initial_indices = make_grids(config, torch.device("cpu"))
    mu_pool, L_pool = get_distribution_parameters(x_pool, config)
    mu_test, L_test = get_distribution_parameters(x_test, config)
    points = x_pool.tolist()
    model = GaussianTransportNet(config.hidden_dim)
    round_rng = torch.get_rng_state()
    histories = {policy: tuple(initial_indices) for policy in POLICIES}
    metrics, energy = [], []
    n_fits = 0

    for round_idx in range(config.n_rounds + 1):
        cache = {}
        next_rng = None
        for (protocol, arm), indices in histories.items():
            if indices not in cache:
                torch.set_rng_state(round_rng)
                train_idx = list(indices)
                train_model(model, x_pool[train_idx], mu_pool[train_idx], L_pool[train_idx],
                            config, verbose=False)
                state = torch.get_rng_state()
                if next_rng is not None and not torch.equal(state, next_rng):
                    raise RuntimeError("Training RNG diverged between policies")
                next_rng = state
                cache[indices] = (evaluate(model, x_test, mu_test, L_test), gaussian_kernels(model, x_pool))
                n_fits += 1
            error, kernels = cache[indices]
            acquired = indices[-config.batch_size:] if round_idx else indices
            x1, x2 = zip(*(points[i] for i in acquired))
            metrics.append([
                seed, round_idx, protocol, arm, len(indices), error["total"], error["mean"],
                error["covariance"], list(acquired), x1, x2,
            ])
            if protocol == "original" and arm in TRANSPORT_ARMS:
                energy.extend(energy_rows(seed, round_idx, arm, points, indices, kernels, config))
            if round_idx == config.n_rounds:
                continue
            if arm == "random":
                rng = np.random.RandomState(seed * 100 + round_idx)
                remaining = list(set(range(len(points))) - set(indices))
                batch = rng.choice(remaining, size=config.batch_size, replace=False).tolist()
            else:
                batch = select(kernels[arm], indices, config.batch_size, config.lam, protocol)
            if len(batch) != config.batch_size or set(batch).intersection(indices):
                raise RuntimeError("Acquisition returned an invalid batch")
            histories[protocol, arm] = (*indices, *batch)
        round_rng = next_rng

    return {"seed": seed, "n_fits": n_fits, "metrics": metrics, "energy": energy}


def write_part(seed, *, config, parts):
    data = run_seed(seed, config=config)
    target = parts / f"{seed:04d}.json.gz"
    temporary = target.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as out:
        json.dump(data, out, separators=(",", ":"))
    temporary.replace(target)
    return seed, data["n_fits"]


def merge_parts(parts, seeds, output):
    counts = {}
    with ((output / "active_learning.csv").open("w", newline="", encoding="utf-8") as active,
          gzip.open(output / "energy_pool.csv.gz", "wt", newline="", encoding="utf-8") as energy):
        active_writer = csv.writer(active, lineterminator="\n")
        energy_writer = csv.writer(energy, lineterminator="\n")
        active_writer.writerow(METRIC_COLUMNS)
        energy_writer.writerow(ENERGY_COLUMNS)
        for seed in seeds:
            with gzip.open(parts / f"{seed:04d}.json.gz", "rt", encoding="utf-8") as file:
                data = json.load(file)
            counts[seed] = data["n_fits"]
            for row in data["metrics"]:
                active_writer.writerow((*row[:-3], *(json.dumps(item) for item in row[-3:])))
            energy_writer.writerows(data["energy"])
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int, help="Override steps for a smoke run only")
    args = parser.parse_args()
    if args.seeds < 1 or args.jobs < 1 or (args.smoke_steps is not None and args.smoke_steps < 1):
        parser.error("Seeds, jobs, and smoke steps must be positive")
    config = replace(load_config("both_complex_b2"), device="cpu")
    if args.smoke_steps is not None:
        config = replace(config, n_train_steps=args.smoke_steps)
    output = args.output_dir.resolve()
    parts = output / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    metadata = {
        "config": asdict(config), "seeds": seeds, "protocols": {
            name: {"normalization": "unit_mean_diagonal" if normalized else "none",
                   "observed_lam": config.lam, "hypothetical_lam": config.lam if coherent else 1.0}
            for name, (normalized, coherent) in PROTOCOLS.items()
        },
        "policies": [list(policy) for policy in POLICIES],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "architecture": platform.machine(), "torch_threads": 1},
    }
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        parser.error("Output contains a different run configuration")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    pending = [i for i in seeds if not (parts / f"{i:04d}.json.gz").exists()]
    write = partial(write_part, config=config, parts=parts)
    start = time.monotonic()
    torch.set_num_threads(1)
    if args.jobs == 1 or len(pending) <= 1:
        for seed in pending:
            _, n_fits = write(seed)
            print(f"Seed {seed} complete ({n_fits} distinct fits)", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn"),
            initializer=torch.set_num_threads, initargs=(1,),
        ) as executor:
            futures = [executor.submit(write, seed) for seed in pending]
            for done in as_completed(futures):
                seed, n_fits = done.result()
                print(f"Seed {seed} complete ({n_fits} distinct fits, {time.monotonic()-start:.0f}s)", flush=True)
    counts = merge_parts(parts, seeds, output)
    print(f"Saved {len(seeds)} seeds, {sum(counts.values())} distinct fits ({time.monotonic()-start:.0f}s)")


if __name__ == "__main__":
    main()
