"""NeuralOT on the evaluation sets (env_neuralot, repository root).

    python -m baselines.neuralot --root experiments/torot/camera_ready --sets 'ood_paper_*' --per-set all --workers 8

NeuralOT is Neural Optimal Transport (Korotin, Selikhanovych, Burnaev, ICLR 2023), trained separately for every
instance on samples drawn from the two histograms: the routine
third_party/NOT_toy_1D_cost_strong.py (2,500 steps, its default networks and learning rates), samples from
sampler.py:HistogramSampler on [0, 1], and the plan is the 2D histogram of (x, T(x)) on the n-bin grid. All six
cost types of the corpus are implemented.

One instance takes 10-20 minutes, almost all of it single-threaded Python, so several instances run per GPU
(--workers). The routine's last step pushes 16M samples through the map at once (~13 GB); a per-GPU file lock lets only
one worker at a time run it, which changes nothing in the computation. Every finished instance is cached
(<root>/preds_subset/neuralot_cache/<set>/<index>.npz), so runs can be interrupted and resumed.
Outputs: <root>/preds_subset/<set>__neuralot.npz (idx, plans, seconds) for the instances done; with all instances of a
set, also <root>/preds/<set>__neuralot.npy (+ .timing.json, run times not comparable: several per GPU).

2D sets (reproduce/build_sets_2d.py): the 2D routine third_party/NOT_toy_2D_cost_strong.py (same
steps and networks), samples drawn from the 2D histogram (uniform within each bin) on the grid rescaled to
[-2.5, 2.5]^2, where that routine bins (x, T(x)) into the side x side grid (x-major, the order of the sets). Rescaling
does not change the OT problem for |x-y|^2 and |x-y| (max-normalised costs); the routine has no other corpus cost.
"""
from __future__ import annotations

import argparse
import fcntl
import multiprocessing as mp
import os
import time

import numpy as np

from baselines.common import find_sets, load_set, provenance, save_predictions, write_json

COSTS = ("sqeuclidean", "euclidean", "canberra", "sqrt_euclidean", "cubic_euclidean", "gaussian")
COSTS_2D = ("sqeuclidean", "euclidean")      # the corpus costs that the 2D routine implements


class HistogramSampler2D:
    """Points from a 2D histogram on a side x side grid of [-2.5, 2.5]^2 (x-major): a bin drawn with its mass, then a
    uniform point inside it."""

    def __init__(self, weights, side, device):
        import torch
        self.w = torch.as_tensor(weights, dtype=torch.float32, device=device)
        c = -2.5 + 5.0 * (torch.arange(side, dtype=torch.float32, device=device) + 0.5) / side
        self.centers = torch.stack(torch.meshgrid(c, c, indexing="ij"), -1).reshape(-1, 2)
        self.width, self.device = 5.0 / side, device

    def sample(self, n):
        import torch
        k = torch.multinomial(self.w, n, replacement=True)
        return self.centers[k] + (torch.rand(n, 2, device=self.device) - 0.5) * self.width


def _locked(fn, lock_path):
    """fn wrapped in an exclusive file lock (one memory-heavy call per GPU at a time); the memory it used is handed
    back to the GPU before the lock is released (PyTorch would otherwise keep it cached in this process)."""
    import torch

    def wrapper(*args, **kwargs):
        with open(lock_path, "w") as fp:
            fcntl.flock(fp, fcntl.LOCK_EX)
            try:
                return fn(*args, **kwargs)
            finally:
                torch.cuda.empty_cache()
                fcntl.flock(fp, fcntl.LOCK_UN)
    return wrapper


def solve(job):
    """One instance: (set path, index, steps, cache file, lock file) -> (index, seconds). Runs in a worker process."""
    import torch
    from baselines.neuralot.sampler import HistogramSampler
    from baselines.neuralot.third_party import NOT_toy_1D_cost_strong as NOT
    path, k, steps, cache, lock = job
    s = load_set(path)
    dev = torch.device("cuda")
    if s["dim"] == 2:
        from baselines.neuralot.third_party import NOT_toy_2D_cost_strong as NOT2
        if not getattr(NOT2.plot_generated_2D, "_locked", False):
            NOT2.plot_generated_2D = _locked(NOT2.plot_generated_2D, lock); NOT2.plot_generated_2D._locked = True
        side = int(round(np.sqrt(s["n"])))
        X, Y = HistogramSampler2D(s["a"][k], side, dev), HistogramSampler2D(s["b"][k], side, dev)
        t0 = time.perf_counter()
        with torch.enable_grad():
            plan = NOT2.main_routine(X, Y, side, rng=None, metric=s["cost"], max_steps=steps, verbose=False)
        seconds = time.perf_counter() - t0
        plan = plan[-1] if isinstance(plan, tuple) else plan
        np.savez(cache, plan=np.asarray(plan, dtype=np.float64), seconds=seconds)
        return k, seconds
    if not getattr(NOT.plot_1D, "_locked", False):
        NOT.plot_1D = _locked(NOT.plot_1D, lock); NOT.plot_1D._locked = True
    X = HistogramSampler(torch.as_tensor(s["a"][k], dtype=torch.float32, device=dev), ranges=(0, 1), device=dev)
    Y = HistogramSampler(torch.as_tensor(s["b"][k], dtype=torch.float32, device=dev), ranges=(0, 1), device=dev)
    t0 = time.perf_counter()
    with torch.enable_grad():
        _, plan = NOT.main_routine(X, Y, num_bins=s["n"], rng=(0, 1), metric=s["cost"], max_steps=steps, verbose=False)
    seconds = time.perf_counter() - t0
    np.savez(cache, plan=np.asarray(plan, dtype=np.float64), seconds=seconds)
    return k, seconds


def nested_subset(N: int, size: int, base: int = 100) -> np.ndarray:
    """Evenly spaced instance indices; larger subsets contain the smaller ones (10 within 100 within e.g. 500), so
    instances already solved for a smaller subset are reused from the cache."""
    grid = lambda k: np.unique(np.linspace(0, N - 1, k).round().astype(int))
    if size <= base:
        return grid(size)
    idx = grid(base)
    rest = np.setdiff1d(np.arange(N), idx)
    extra = rest[np.unique(np.linspace(0, len(rest) - 1, min(size - len(idx), len(rest))).round().astype(int))]
    return np.sort(np.concatenate([idx, extra]))


def collect(root: str, path: str, steps: int, workers: int) -> None:
    s = load_set(path)
    cache_dir = f"{root}/preds_subset/neuralot_cache/{s['name']}"
    done = sorted(int(f[:-4]) for f in os.listdir(cache_dir) if f.endswith(".npz")) if os.path.isdir(cache_dir) else []
    if not done:
        return
    plans = [np.load(f"{cache_dir}/{k}.npz") for k in done]
    idx, P, sec = np.array(done), np.stack([z["plan"] for z in plans]), np.array([float(z["seconds"]) for z in plans])
    np.savez(f"{root}/preds_subset/{s['name']}__neuralot.npz", idx=idx, plans=P, seconds=sec)
    info = {"set": s["name"], "instances": len(idx), "steps": steps, "parallel_workers_per_gpu": workers,
            "note": "run times not comparable (several instances per GPU in parallel)", "provenance": provenance()}
    write_json(f"{root}/preds_subset/{s['name']}__neuralot.json", {"method": "neuralot", **info, "indices": idx.tolist()})
    if len(idx) == len(s["a"]):                                   # complete set: score it like every other method
        save_predictions(root, s["name"], "neuralot", P, {**info, "device": "GPU (several workers)",
                         "single_instance_s_median": float(np.median(sec)), "batched_s_per_instance": float(np.median(sec))})
    print(f"neuralot {s['name']}: {len(idx)}/{len(s['a'])} instances, median {np.median(sec) / 60:.1f} min each", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--sets", nargs="+", default=["id_simplex_*", "ood_paper_*"])
    ap.add_argument("--per-set", default="10", help="instances per set (evenly spaced), or 'all'")
    ap.add_argument("--workers", type=int, default=8, help="instances in parallel per GPU")
    ap.add_argument("--steps", type=int, default=2500, help="NeuralOT training steps (the paper: 2500)")
    ap.add_argument("--shard", default="0/1", help="i/k: only instances with index %% k == i (split a set over GPUs)")
    args = ap.parse_args()
    paths = [p for p in find_sets(args.root, args.sets)
             if load_set(p)["cost"] in (COSTS_2D if load_set(p)["dim"] == 2 else COSTS)]
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").replace(",", "_")
    lock = f"{args.root}/preds_subset/.neuralot_gpu{gpu}.lock"
    os.makedirs(f"{args.root}/preds_subset", exist_ok=True)
    jobs = []
    for path in paths:
        s = load_set(path)
        cache_dir = f"{args.root}/preds_subset/neuralot_cache/{s['name']}"
        os.makedirs(cache_dir, exist_ok=True)
        N = len(s["a"])
        idx = np.arange(N) if args.per_set == "all" else nested_subset(N, int(args.per_set))
        shard, shards = map(int, args.shard.split("/"))
        jobs += [(path, int(k), args.steps, f"{cache_dir}/{k}.npz", lock) for k in idx
                 if k % shards == shard and not os.path.exists(f"{cache_dir}/{k}.npz")]
    print(f"{len(jobs)} instances to solve on GPU {gpu} with {args.workers} workers", flush=True)
    ctx = mp.get_context("spawn")                                 # CUDA in worker processes
    with ctx.Pool(args.workers, maxtasksperchild=20) as pool:
        for n, (k, sec) in enumerate(pool.imap_unordered(solve, jobs), 1):
            if n % 10 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} done (last: {sec / 60:.1f} min)", flush=True)
    for path in paths:
        collect(args.root, path, args.steps, args.workers)


if __name__ == "__main__":
    main()
