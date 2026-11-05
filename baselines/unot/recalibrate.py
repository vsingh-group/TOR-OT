"""Finalise UNOT runs trained before BatchNorm recalibration was part of train.py (env_unot, repository root).

    python -m baselines.unot.recalibrate --runs 'experiments/baselines/unot/*' --val experiments/torot/camera_ready

For each run: keeps the official loop's predictor as predictor_trained.pt, recalibrates the BatchNorm running
statistics on the trained generator (train.recalibrate_batchnorm), writes predictor.pt and recomputes metrics.json.
"""
import argparse
import glob
import json
import os

import torch

from baselines.common import find_sets
from baselines.unot.train import build, finalize

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--val", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run in sorted({r for pat in args.runs for r in glob.glob(pat) if os.path.exists(f"{r}/config.json")}):
        meta = json.load(open(f"{run}/config.json"))
        if not os.path.exists(f"{run}/predictor_trained.pt"):
            os.replace(f"{run}/predictor.pt", f"{run}/predictor_trained.pt")
        generator, predictor = build(meta["n"], meta["hyperparameters"], device, meta.get("dim", 1))
        predictor.load_state_dict(torch.load(f"{run}/predictor_trained.pt", map_location=device, weights_only=True))
        generator.load_state_dict(torch.load(f"{run}/generator.pt", map_location=device, weights_only=True))
        previous = json.load(open(f"{run}/metrics.json"))
        print(run, end=": ", flush=True)
        finalize(run, predictor, generator, find_sets(args.val, [f"val_simplex_{meta['cost']}"], folder="sets_val")[0], device,
                 {"train_seconds": previous.get("train_seconds")})
