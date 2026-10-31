"""Decoding latency and throughput as a function of the support size n.

    python -m torot.benchmark --checkpoint experiments/torot/v2_25ep/best.pt \\
        --sizes 5 10 20 30 50 75 100 150 200 --out latency.json

For every n, instances with exactly n bins are taken from ``test_id`` (n <= 100) or
``test_extrap`` (n > 100). Reported per n:

* single-instance latency (batch of 1, median over ``--repeats`` instances),
* batched throughput (instances per second at ``--batch_size``),
* tokens per plan of the reference serialisation (plan format),
* POT's exact solver time on the same instances (CPU), for reference.

GPU timings are synchronised; every size is warmed up once before timing.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import ot
import torch

from torot.data.cost import COST_TYPES, cost_matrix
from torot.data.dataset import OTPlanDataset, load_corpus
from torot.decode import predict_plans
from torot.evaluate import load_model


def _timed(fn, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def instances_of_size(corpus, n: int, limit: int) -> list:
    split = corpus["test_id"] if n <= 100 else corpus["test_extrap"]
    idx = np.nonzero(np.asarray(split["n"]) == n)[0][:limit]
    data = OTPlanDataset(split)
    return [data[int(i)] for i in idx]


def pot_time(inst: dict) -> float:
    C = cost_matrix(inst["coords_src"][:, : inst["dim"]], inst["coords_tgt"][:, : inst["dim"]], COST_TYPES[inst["cost_type"]])
    t0 = time.perf_counter()
    ot.emd(inst["a"], inst["b"], C)
    return time.perf_counter() - t0


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--repo_id", default=None, help="corpus to draw instances from (default: the checkpoint's)")
    p.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 20, 30, 50, 75, 100, 150, 200])
    p.add_argument("--repeats", type=int, default=8, help="instances timed one at a time per size")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_len", type=int, default=4000)
    p.add_argument("--last_row", choices=("remainder", "model"), default="remainder",
                   help="row decoding: last row = remaining capacity (default) or decoded by the model")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_model(args.checkpoint, device)
    corpus = load_corpus(args.repo_id or cfg.data.repo_id, cfg.data.cache_dir)

    rows = []
    for n in args.sizes:
        insts = instances_of_size(corpus, n, max(args.repeats, args.batch_size))
        if not insts:
            print(f"n={n}: no instances, skipped")
            continue
        predict = lambda xs: predict_plans(model, tok, xs, device, max_len=args.max_len, last_row=args.last_row)
        predict(insts[:1])                                                  # warm-up
        single = [_timed(lambda x=x: predict([x]), device) for x in insts[: args.repeats]]
        batch = (insts * (args.batch_size // len(insts) + 1))[: args.batch_size]
        batched = _timed(lambda: predict(batch), device)
        tokens = [len(tok.encode(x["plan_rows"], x["plan_cols"], x["plan_vals"], x["n"])[0]) for x in insts]
        row = {
            "n": n,
            "tokens_per_plan": float(np.mean(tokens)),
            "latency_single_s": float(np.median(single)),
            "throughput_inst_per_s": len(batch) / batched,
            "pot_s": float(np.median([pot_time(x) for x in insts[: args.repeats]])),
        }
        rows.append(row)
        print(f"n={n:>4}  tokens/plan {row['tokens_per_plan']:7.0f}  single {row['latency_single_s']:7.3f} s  "
              f"batch{args.batch_size} {row['throughput_inst_per_s']:7.1f} inst/s  POT {row['pot_s'] * 1e3:7.2f} ms", flush=True)

    ns = np.log([r["n"] for r in rows])
    fit = {k: float(np.polyfit(ns, np.log([r[v] for r in rows]), 1)[0])
           for k, v in (("latency_single", "latency_single_s"), ("time_per_inst_batched", "throughput_inst_per_s"), ("tokens", "tokens_per_plan"))}
    fit["time_per_inst_batched"] = -fit["time_per_inst_batched"]           # time = 1 / throughput
    print("log-log slopes vs n (1 = linear, 2 = quadratic):", {k: round(v, 2) for k, v in fit.items()})
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "decoding": cfg.model.decoding, "device": str(device),
                       "batch_size": args.batch_size, "sizes": rows, "loglog_slope": fit}, f, indent=2)


if __name__ == "__main__":
    main()
