"""Reconstruct benchmark checkpoints from their recorded ordered acquisitions."""

import argparse
import csv
import hashlib
import json
import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

from train.experiment import (
    ExperimentConfig,
    GaussianTransportNet,
    evaluate,
    get_distribution_parameters,
    make_grids,
    train_model,
)
from train.gaussian_validation import TRANSPORT_ARMS, gaussian_kernels, select


DEFAULT_HISTORICAL_DIR = Path("results/gaussian_2d/validation/active_learning")
METRICS = ("total", "mean", "covariance")


def read_history(path):
    histories = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            arm = row["arm"]
            if row["protocol"] != "original" or arm not in TRANSPORT_ARMS:
                continue
            seed, round_idx = int(row["seed"]), int(row["round"])
            history = histories.setdefault(seed, {}).setdefault(arm, {})
            if round_idx in history:
                raise ValueError(f"Duplicate historical record for {(seed, arm, round_idx)}")
            history[round_idx] = {
                "batch": json.loads(row["acquired_indices"]),
                "budget": int(row["budget"]),
                "test_metrics": {metric: float(row[metric]) for metric in METRICS},
            }
    return histories


def refit_seed(seed, *, config, history, max_round):
    """Return exact CPU replays with each recorded acquisition checked on its own model."""
    torch.manual_seed(seed)
    cpu = torch.device("cpu")
    x_pool, x_test, initial = make_grids(config, cpu)
    mu_pool, factor_pool = get_distribution_parameters(x_pool, config)
    mu_test, factor_test = get_distribution_parameters(x_test, config)
    model = GaussianTransportNet(config.hidden_dim)
    round_rng = torch.get_rng_state()
    checkpoints, metric_checks, acquisition_checks, fits = [], 0, 0, 0
    labels = {arm: [] for arm in TRANSPORT_ARMS}

    for round_idx in range(max_round + 1):
        cache = {}
        next_rng = None
        for arm in TRANSPORT_ARMS:
            record = history[arm][round_idx]
            labels[arm].extend(record["batch"])
            ordered = tuple(labels[arm])
            context = f"seed {seed}, round {round_idx}, {arm}"
            if (list(ordered[:len(initial)]) != initial or len(ordered) != record["budget"]
                    or len(ordered) != len(set(ordered))):
                raise ValueError(f"Invalid historical labels for {context}")
            expected_batch = []
            if round_idx < config.n_rounds:
                expected_batch = history[arm][round_idx + 1]["batch"]

            if ordered not in cache:
                torch.set_rng_state(round_rng)
                selected = list(ordered)
                train_model(model, x_pool[selected], mu_pool[selected], factor_pool[selected],
                            config, verbose=False)
                state = torch.get_rng_state()
                if next_rng is not None and not torch.equal(state, next_rng):
                    raise RuntimeError(f"Training RNG mismatch for {context}")
                next_rng = state
                metrics = evaluate(model, x_test, mu_test, factor_test)
                weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
                kernels = gaussian_kernels(model, x_pool) if expected_batch else {}
                cache[ordered] = (metrics, weights, kernels)
                fits += 1
            metrics, weights, kernels = cache[ordered]
            if metrics != record["test_metrics"]:
                raise RuntimeError(f"Historical test metrics differ for {context}. "
                                   f"Expected {record['test_metrics']}, computed {metrics}")
            metric_checks += len(METRICS)
            batches = {}
            if expected_batch:
                batch = select(kernels[arm], ordered, config.batch_size, config.lam, "original")
                if batch != expected_batch:
                    raise RuntimeError(f"Historical acquisition differs for {context}. "
                                       f"Expected {expected_batch}, computed {batch}")
                acquisition_checks += 1
                batches = {arm: batch}

            if round_idx == 0 and arm != TRANSPORT_ARMS[0]:
                checkpoints[0]["historical_next_batches"].update(batches)
            else:
                checkpoints.append({
                    "round": round_idx,
                    "trajectory": "shared" if round_idx == 0 else arm,
                    "ordered_labels": list(ordered),
                    "state_dict": weights,
                    "test_metrics": metrics,
                    "historical_next_batches": batches,
                })
        round_rng = next_rng

    return {
        "seed": seed,
        "config": asdict(config),
        "checkpoints": checkpoints,
        "verification": {
            "exact_historical_metric_values": metric_checks,
            "exact_historical_ordered_batches": acquisition_checks,
            "distinct_fits": fits,
        },
    }


def checkpoint_path(output, seed):
    return output / f"seed_{seed:04d}.pt"


def write_seed(seed, *, config, histories, max_round, output):
    target = checkpoint_path(output, seed)
    temporary = target.with_suffix(".tmp")
    data = refit_seed(seed, config=config, history=histories[seed], max_round=max_round)
    torch.save(data, temporary)
    temporary.replace(target)
    return seed, data["verification"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-dir", type=Path, default=DEFAULT_HISTORICAL_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", help="Replay a subset of the archived seed IDs")
    parser.add_argument("--max-round", type=int, help="Last checkpoint to reconstruct, inclusive")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    historical_csv = args.historical_dir / "active_learning.csv"
    historical_manifest = args.historical_dir / "manifest.json"
    original = json.loads(historical_manifest.read_text(encoding="utf-8"))
    config = ExperimentConfig(**original["config"])
    histories = read_history(historical_csv)
    seeds = sorted(set(args.seeds if args.seeds is not None else original["seeds"]))
    max_round = config.n_rounds if args.max_round is None else args.max_round
    if not 0 <= max_round <= config.n_rounds or not set(seeds) <= set(histories):
        parser.error("Invalid round or seed IDs for the archived campaign")
    if config.device != "cpu":
        parser.error("This replay targets the archived CPU campaign")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "manifest.json"
    metadata = {
        "format_version": 1,
        "historical_source": {
            "active_learning_csv_sha256": sha256(historical_csv),
            "manifest_sha256": sha256(historical_manifest),
        },
        "config": asdict(config),
        "max_round": max_round,
        "trajectories": list(TRANSPORT_ARMS),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "architecture": platform.machine(), "torch_threads": 1,
        },
    }
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if previous["metadata"] != metadata:
            parser.error("Output directory contains checkpoints from a different replay")
    provenance_path.write_text(json.dumps({"metadata": metadata}, indent=2) + "\n", encoding="utf-8")

    pending = [seed for seed in seeds if not checkpoint_path(output, seed).exists()]
    write = partial(write_seed, config=config, histories={seed: histories[seed] for seed in pending},
                    max_round=max_round, output=output)
    torch.set_num_threads(1)
    if args.jobs > 1 and len(pending) > 1:
        with ProcessPoolExecutor(
            max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn"),
            initializer=torch.set_num_threads, initargs=(1,),
        ) as executor:
            futures = [executor.submit(write, seed) for seed in pending]
            for future in as_completed(futures):
                seed, verification = future.result()
                print(f"Seed {seed} complete. Exact checks {verification}", flush=True)
    else:
        for seed in pending:
            seed, verification = write(seed)
            print(f"Seed {seed} complete. Exact checks {verification}", flush=True)

    verification_by_seed = {}
    counts = {}
    for file in sorted(output.glob("seed_*.pt")):
        data = torch.load(file, map_location="cpu", weights_only=True)
        verification_by_seed[data["seed"]] = data["verification"]
        counts[data["seed"]] = len(data["checkpoints"])
    result = {"metadata": metadata, "checkpoint_counts": counts,
              "verification_by_seed": verification_by_seed}
    provenance_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {sum(counts.values())} reconstructed states from {len(counts)} seeds")


if __name__ == "__main__":
    main()
