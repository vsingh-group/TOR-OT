"""Evaluate a checkpoint on a dataset split.

    python -m torot.evaluate --checkpoint experiments/torot/base/best.pt --split test_id \\
        --max_instances 2000 --symmetric --out results_test_id.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Optional

import torch
from datasets import load_dataset

from torot.config import load_config
from torot.data.dataset import OTPlanDataset
from torot.decode import predict_plans
from torot.metrics import aggregate, evaluate_instance
from torot.model import ToROT
from torot.tokenizer import PlanTokenizer


def load_model(checkpoint: str, device: torch.device, use_ema: bool = True) -> tuple[ToROT, PlanTokenizer, object]:
    cfg = load_config(os.path.join(os.path.dirname(checkpoint), "config.yaml"))
    tok = PlanTokenizer(cfg.tokenizer.mantissa_digits, cfg.tokenizer.max_exponent)
    model = ToROT(cfg.model, tok.vocab_size).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["ema" if use_ema else "model"])
    model.eval()
    return model, tok, cfg


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test_id")
    p.add_argument("--repo_id", default=None, help="override the dataset repo of the config")
    p.add_argument("--max_instances", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--symmetric", action="store_true", help="transpose-averaged decoding (symmetry 6.1)")
    p.add_argument("--raw_weights", action="store_true", help="use the raw instead of the EMA weights")
    p.add_argument("--out", default=None, help="JSON file for aggregated and per-instance metrics")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_model(args.checkpoint, device, use_ema=not args.raw_weights)
    ds = load_dataset(args.repo_id or cfg.data.repo_id, cache_dir=cfg.data.cache_dir)[args.split]
    data = OTPlanDataset(ds)
    total = len(data) if args.max_instances is None else min(len(data), args.max_instances)

    per_instance = []
    latency = []
    for start in range(0, total, args.batch_size):
        instances = [data[i] for i in range(start, min(total, start + args.batch_size))]
        t0 = time.perf_counter()
        plans = predict_plans(model, tok, instances, device, max_len=args.max_len, symmetric=args.symmetric)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency.append((time.perf_counter() - t0) / len(instances))
        for P, inst in zip(plans, instances):
            rec = evaluate_instance(P, inst)
            rec["n"], rec["m"], rec["cost_type"] = inst["n"], inst["m"], inst["cost_type"]
            per_instance.append(rec)

    summary = aggregate(per_instance)
    summary["latency_per_instance_s"] = sum(latency) / len(latency)
    summary["instances"] = total
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_instance": per_instance}, f)


if __name__ == "__main__":
    main()
