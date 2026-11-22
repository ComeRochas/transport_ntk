"""Run the 2D-Gaussian AL study for the configuration in
train/configs/configs_2d_gaussian.yml (`both_complex_b2`, the main-text figure)."""

import torch
import time
import os
import json
import argparse
import yaml
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import partial
from experiment import ExperimentConfig, run_experiment

RESULTS_DIR = "./results"
STRATEGIES = ["mean", "transport", "random"]
AVAILABLE_STRATEGIES = [*STRATEGIES, "projected", "projected_canonical"]


def load_configs(path="train/configs/configs_2d_gaussian.yml"):
    with open(path) as f:
        raw = yaml.safe_load(f)
    return [ExperimentConfig(**entry) for entry in raw]


def run_seed(seed, *, config, strategies, verbose):
    return {strategy: run_experiment(strategy, seed, config, verbose=verbose)
            for strategy in strategies}


def run_config(config, n_seeds, strategies, *, out_root=RESULTS_DIR, jobs=1):
    out_dir = os.path.join(out_root, config.name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    all_results = {s: [] for s in strategies}
    all_indices = {s: [] for s in strategies}

    t0 = time.time()
    run = partial(run_seed, config=config, strategies=strategies, verbose=jobs == 1)

    def collect(seed_results):
        for strategy in strategies:
            res, idx = seed_results[strategy]
            all_results[strategy].append(res)
            all_indices[strategy].append(idx)
        completed = len(all_results[strategies[0]])
        print(f"[{config.name}] Seed {completed}/{n_seeds} complete ({time.time()-t0:.0f}s)", flush=True)

    if jobs == 1:
        for seed in range(n_seeds):
            collect(run(seed))
    else:
        # Each worker runs complete seeds, keeping the RNG independent across seeds.
        with ProcessPoolExecutor(max_workers=jobs, initializer=torch.set_num_threads,
                                 initargs=(torch.get_num_threads(),)) as executor:
            for seed_results in executor.map(run, range(n_seeds)):
                collect(seed_results)

    torch.save({"results": all_results, "indices": all_indices},
               os.path.join(out_dir, "results.pt"))
    print(f"\n[{config.name}] Done in {time.time()-t0:.0f}s → {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--configs", nargs="*", help="Run specific configs by name")
    parser.add_argument("--config-file", default="train/configs/configs_2d_gaussian.yml")
    parser.add_argument("--strategies", nargs="+", choices=AVAILABLE_STRATEGIES, default=STRATEGIES)
    parser.add_argument("--device", help="Override the device in the YAML (e.g. cpu)")
    parser.add_argument("--threads", type=int, help="Number of PyTorch intra-op CPU threads")
    parser.add_argument("--jobs", type=int, default=1, help="Worker processes for independent CPU seeds")
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    args = parser.parse_args()
    if min(args.seeds, args.jobs) < 1 or (args.threads is not None and args.threads < 1):
        parser.error("--seeds, --jobs and --threads must be positive")
    if len(set(args.strategies)) != len(args.strategies):
        parser.error("--strategies must not contain duplicates")
    if args.threads is not None or args.jobs > 1:
        torch.set_num_threads(args.threads or 1)

    configs = load_configs(args.config_file)
    if args.configs:
        configs = [c for c in configs if c.name in args.configs]
    if not configs:
        parser.error("No matching configurations")

    for config in configs:
        if args.device:
            config = replace(config, device=args.device)
        if args.jobs > 1 and config.device != "cpu":
            parser.error("--jobs > 1 requires --device cpu")
        run_config(config, args.seeds, args.strategies, out_root=args.output_dir, jobs=args.jobs)
