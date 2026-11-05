"""Shared helpers for the baselines: problem construction, evaluation sets, metrics and outputs.

numpy + POT only, so that every baseline environment (JAX, PyTorch, CPU) can import it. Run the baselines from the
repository root (``python -m baselines.<method>.<step>``) so that ``torot.data.cost`` is importable from the checkout.

Conventions shared by all baselines (and by ToR-OT's own evaluation in ``reproduce/``):
* a problem is a pair of histograms (a, b) on the n-bin grid of bin centres in [0, 1] (``torot.data.supports.grid_1d``)
  with a max-normalised cost matrix (``torot.data.cost.cost_matrix``, C.max() == 1);
* an evaluation set is ``<root>/sets/<name>.npz`` with arrays a [N, n], b [N, n], plan [N, n, n] (exact, POT) and
  scalars cost, n (written by ``reproduce/build_sets.py``); validation sets live in ``<root>/sets_val/``;
* predictions are ``<root>/preds/<set>__<method>.npy`` (dense plans [N, n, n], float64) with
  ``<set>__<method>.timing.json`` (method, run, device, timings).
"""
from __future__ import annotations

import fnmatch
import glob
import json
import os
import platform
import subprocess
import sys

import numpy as np

from torot.data.cost import cost_matrix as _cost_matrix
import math

from torot.data.supports import grid_1d, grid_2d

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cost_matrix(n: int, cost: str, dim: int = 1) -> np.ndarray:
    """Max-normalised n x n cost matrix of the n-bin grid (identical to the one used to build the sets): 1D (n bins
    on [0, 1]) or 2D (a sqrt(n) x sqrt(n) grid on [0, 1]^2, x-major, ``torot.data.supports.grid_2d``)."""
    x = grid_1d(n) if dim == 1 else grid_2d(math.isqrt(n))
    return _cost_matrix(x, x, cost)


def load_set(path: str) -> dict:
    z = np.load(path)
    return {"name": os.path.basename(path)[:-4], "a": z["a"], "b": z["b"], "plan": z["plan"],
            "cost": str(z["cost"]), "n": int(z["n"]), "dim": int(z["dim"]) if "dim" in z else 1}


def find_sets(root: str, patterns: list[str], cost: str | None = None, folder: str = "sets") -> list[str]:
    """Set files under <root>/<folder> whose name matches one of the fnmatch patterns (and whose cost is ``cost``)."""
    paths = [p for p in sorted(glob.glob(f"{root}/{folder}/*.npz"))
             if any(fnmatch.fnmatch(os.path.basename(p)[:-4], pat) for pat in patterns)]
    return [p for p in paths if cost is None or str(np.load(p)["cost"]) == cost]


def paper_mape(R: np.ndarray, P: np.ndarray, eps: float = 1e-12) -> float:
    """The paper's MAPE (= weighted MAPE): sum |R - P| / sum |R| over the union of both supports."""
    idx = np.union1d(np.flatnonzero(R), np.flatnonzero(P))
    if idx.size == 0:
        return 0.0
    r, p = R.ravel()[idx], P.ravel()[idx]
    r = np.where(r == 0.0, eps, r)
    return float(np.abs(r - p).sum() / np.abs(r).sum())


def plan_summary(plans: np.ndarray, s: dict) -> dict:
    """Validation summary of predicted plans against a set's exact plans."""
    C = cost_matrix(s["n"], s["cost"], s.get("dim", 1))
    mape = np.array([paper_mape(R, P) for R, P in zip(s["plan"], plans)])
    row = np.abs(plans.sum(2) - s["a"]).sum(1)
    col = np.abs(plans.sum(1) - s["b"]).sum(1)
    gap = (plans * C).sum((1, 2)) - (s["plan"] * C).sum((1, 2))
    return {"set": s["name"], "instances": int(len(plans)), "paper_mape_mean": float(mape.mean()),
            "paper_mape_median": float(np.median(mape)), "row_violation_l1_mean": float(row.mean()),
            "col_violation_l1_mean": float(col.mean()), "abs_cost_gap_mean": float(np.abs(gap).mean()),
            "finite_fraction": float(np.isfinite(plans).all((1, 2)).mean())}


def save_predictions(root: str, set_name: str, method: str, plans: np.ndarray, info: dict) -> None:
    os.makedirs(f"{root}/preds", exist_ok=True)
    np.save(f"{root}/preds/{set_name}__{method}.npy", np.asarray(plans, dtype=np.float64))
    write_json(f"{root}/preds/{set_name}__{method}.timing.json", {"method": method, **info})


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=2, default=str)


def provenance() -> dict:
    """Code and interpreter provenance recorded next to every trained baseline and prediction."""
    def git(*args):
        try:
            return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {"tor_ot_commit": git("rev-parse", "HEAD"), "tor_ot_dirty": bool(git("status", "--porcelain", "--", "baselines")),
            "python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform()}


def select_run(runs: list[str], cost: str) -> tuple[str, dict]:
    """Among trained runs for ``cost`` (dirs with config.json + metrics.json), the one with the lowest validation MAPE."""
    cands = {}
    for run in runs:
        cfg = json.load(open(f"{run}/config.json"))
        if cfg["cost"] != cost:
            continue
        if not os.path.exists(f"{run}/metrics.json"):   # never select among an incomplete set of candidates
            raise RuntimeError(f"run {run} for cost {cost!r} has not finished")
        cands[run] = json.load(open(f"{run}/metrics.json"))["validation"]["paper_mape_mean"]
    if not cands:
        raise FileNotFoundError(f"no trained run for cost {cost!r} among {runs}")
    best = min(cands, key=lambda r: (np.nan_to_num(cands[r], nan=np.inf), r))
    return best, {"selected_run": best, "selected_by": "lowest validation paper MAPE (mean)", "candidates": cands}


def predict_sets(method: str, root: str, runs: list[str], patterns: list[str], load, predict, device: str,
                 note: str | None = None) -> None:
    """Predict every set under <root>/sets matching ``patterns`` with the run trained on the set's cost (lowest
    validation MAPE among ``runs``), timing all instances in one batch and 10 single instances (after warm-up).

    ``load(run) -> model`` and ``predict(model, a, b) -> plans [N, n, n] (numpy, synchronised)`` are method-specific.
    """
    import time
    models = {}
    for path in find_sets(root, patterns):
        s = load_set(path)
        try:
            run, selection = select_run(runs, s["cost"])
        except (FileNotFoundError, RuntimeError) as err:
            print(f"{method} {s['name']}: skipped ({err})", flush=True)
            continue
        if run not in models:
            models[run] = load(run)
        model = models[run]
        predict(model, s["a"], s["b"])                                     # warm-up / compilation at this shape
        t0 = time.perf_counter()
        plans = predict(model, s["a"], s["b"])
        batched = (time.perf_counter() - t0) / len(plans)
        predict(model, s["a"][:1], s["b"][:1])
        single = []
        for k in range(10):
            t0 = time.perf_counter()
            predict(model, s["a"][k:k + 1], s["b"][k:k + 1])
            single.append(time.perf_counter() - t0)
        meta = json.load(open(f"{run}/config.json"))
        info = {**selection, "eps": meta["eps"], "device": device, "batched_s_per_instance": batched,
                "batch_instances": len(plans), "single_instance_s_median": float(np.median(single)),
                "provenance": provenance()}
        if note:
            info["note"] = note
        save_predictions(root, s["name"], method, plans, info)
        summ = plan_summary(plans, s)
        print(f"{method} {s['name']}: eps={meta['eps']:g}  paper MAPE {summ['paper_mape_mean']:.4f}  "
              f"{batched * 1e3:.2f} ms/instance batched, {np.median(single) * 1e3:.1f} ms single", flush=True)


def predict_cli(description: str):
    """Arguments shared by the baselines' predict.py entry points."""
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--root", required=True, help="results folder (sets/, preds/)")
    ap.add_argument("--runs", nargs="+", required=True, help="trained run folders (glob patterns allowed)")
    ap.add_argument("--sets", nargs="+", default=["*"], help="fnmatch patterns on set names")
    ap.add_argument("--note", default=None, help="free-text note stored with the timings (e.g. shared GPU)")
    ap.add_argument("--method", default=None, help="name of the outputs (default: the method's name)")
    args = ap.parse_args()
    args.runs = sorted({r for pat in args.runs for r in glob.glob(pat) if os.path.exists(f"{r}/config.json")})
    return args


REFINE_KS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)


def refine_sets(method: str, root: str, runs: list[str], patterns: list[str], load, init, iterate, plan, sync,
                device: str, ks=REFINE_KS, cold: bool = True) -> None:
    """Accuracy and time of a learned initialisation followed by k Sinkhorn iterations (the methods' intended use).

    For every set matching ``patterns`` (run chosen on validation as in ``predict_sets``):
    ``init(model, a, b, warm) -> state`` gives the k = 0 state (warm: the network's prediction; cold: Sinkhorn's
    standard start), ``iterate(model, state, a, b, k) -> state`` runs k full iterations, ``plan(model, state)`` returns
    the plans on the device and ``sync(plans) -> numpy`` waits for the device and copies them back. Accuracy uses the iterates of one pass; times are measured from
    scratch for every k (all instances in one batch, and the median of 5 single instances).
    Writes <root>/warmstart/<set>__<method>.json (and __sinkhorn_cold_eps<eps>.json with ``cold``).
    """
    import time
    for path in find_sets(root, patterns):
        s = load_set(path)
        try:
            run, selection = select_run(runs, s["cost"])
        except (FileNotFoundError, RuntimeError) as err:
            print(f"{method} {s['name']}: skipped ({err})", flush=True)
            continue
        model, eps = load(run), json.load(open(f"{run}/config.json"))["eps"]
        a, b = s["a"], s["b"]
        for warm in ((True, False) if cold else (True,)):
            name = method if warm else f"sinkhorn_cold_eps{eps:g}"
            rows, state, done_k = [], init(model, a, b, warm), 0
            for k in ks:
                state = iterate(model, state, a, b, k - done_k); done_k = k
                summ = plan_summary(sync(plan(model, state)), s)

                def full(aa, bb, k=k):
                    return sync(plan(model, iterate(model, init(model, aa, bb, warm), aa, bb, k)))
                full(a, b); full(a[:1], b[:1])                                  # compile / warm up these shapes
                t0 = time.perf_counter(); full(a, b); batched = (time.perf_counter() - t0) / len(a)
                single = []
                for j in range(5):
                    t0 = time.perf_counter(); full(a[j:j + 1], b[j:j + 1]); single.append(time.perf_counter() - t0)
                rows.append({"k": k, **{key: summ[key] for key in ("paper_mape_mean", "paper_mape_median", "abs_cost_gap_mean",
                                                                   "row_violation_l1_mean", "col_violation_l1_mean", "finite_fraction")},
                             "batched_s_per_instance": batched, "single_instance_s_median": float(np.median(single))})
            write_json(f"{root}/warmstart/{s['name']}__{name}.json",
                       {"method": name, "eps": eps, **({"init": "learned"} if warm else {"init": "cold (Sinkhorn's standard start)"}),
                        **selection, "device": device, "provenance": provenance(), "by_k": rows})
            print(f"{name:24s} {s['name']:30s} MAPE k=0 {rows[0]['paper_mape_mean']:.3f} -> k={ks[-1]} {rows[-1]['paper_mape_mean']:.3f}", flush=True)
