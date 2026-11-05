"""Entropic OT with Sinkhorn on the GPU, batched (env_torot, repository root): the GPU counterpart of baselines/pot/solve.py.

    python -m baselines.pot.sinkhorn_gpu --root experiments/torot/camera_ready --sets 'id_simplex_*' 'ood_paper_*'

Log-domain Sinkhorn in float64 on all instances of a set at once, from the standard start (zero potentials), at eps =
0.01 and 0.001 (absolute; the cost max is 1), until the L1 marginal error of every instance is below --tol (checked every
10 iterations) or --max-iter. Records the iterations needed and the time per instance: all instances in one batch, and
the median of 5 single instances (each after a warm-up). Writes <root>/preds/<set>__sinkhorn_gpu_eps<eps>.npy (+ timing).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from baselines.common import cost_matrix, find_sets, load_set, plan_summary, provenance, save_predictions


def sinkhorn(a: torch.Tensor, b: torch.Tensor, C: torch.Tensor, eps: float, tol: float, max_iter: int):
    """Batched log-domain Sinkhorn; returns plans [B, n, m] and the iterations run."""
    la, lb = torch.log(a), torch.log(b)
    f, g = torch.zeros_like(a), torch.zeros_like(b)
    M = -C / eps
    it = 0
    while it < max_iter:
        for _ in range(10):
            g = eps * (lb - torch.logsumexp(M[None] + f[:, :, None] / eps, dim=1))
            f = eps * (la - torch.logsumexp(M[None] + g[:, None, :] / eps, dim=2))
        it += 10
        P = torch.exp(M[None] + (f[:, :, None] + g[:, None, :]) / eps)
        err = (P.sum(1) - b).abs().sum(1)                      # column error (rows are exact after the f-update)
        if float(err.max()) < tol:
            break
    return P, it


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--sets", nargs="+", default=["*"])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.001])
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--max-iter", type=int, default=100_000)
    args = ap.parse_args()
    dev = torch.device("cuda")

    def t(x):
        return torch.as_tensor(x, dtype=torch.float64, device=dev)

    for path in find_sets(args.root, args.sets):
        s = load_set(path)
        C = t(cost_matrix(s["n"], s["cost"], s["dim"]))
        a, b = t(np.maximum(s["a"], 1e-300)), t(np.maximum(s["b"], 1e-300))   # empty bins: tiny mass instead of log(0)
        for eps in args.eps:
            run = lambda aa, bb: sinkhorn(aa, bb, C, eps, args.tol, args.max_iter)
            run(a[:2], b[:2]); torch.cuda.synchronize()
            t0 = time.perf_counter(); P, iters = run(a, b); torch.cuda.synchronize()
            batched = (time.perf_counter() - t0) / len(a)
            single, single_iters = [], []
            for k in range(5):
                run(a[k:k + 1], b[k:k + 1]); torch.cuda.synchronize()
                t0 = time.perf_counter(); _, it_k = run(a[k:k + 1], b[k:k + 1]); torch.cuda.synchronize()
                single.append(time.perf_counter() - t0); single_iters.append(it_k)
            plans = P.cpu().numpy()
            method = f"sinkhorn_gpu_eps{eps:g}"
            summ = plan_summary(plans, s)
            save_predictions(args.root, s["name"], method, plans, {
                "eps": eps, "tol_l1_marginal": args.tol, "iterations_batched": iters, "iterations_single_median": float(np.median(single_iters)),
                "device": torch.cuda.get_device_name(), "batched_s_per_instance": batched, "batch_instances": len(a),
                "single_instance_s_median": float(np.median(single)), "provenance": provenance()})
            print(f"{method} {s['name']}: {iters} iterations, MAPE {summ['paper_mape_mean']:.4f}, "
                  f"{batched * 1e3:.3f} ms/instance batched, {np.median(single) * 1e3:.1f} ms single", flush=True)


if __name__ == "__main__":
    main()
