"""Train Meta OT on 1D histograms with the official code (run in env_metaot, from the repository root).

    python -m baselines.metaot.train --cost sqeuclidean --eps 0.01 \\
        --out experiments/baselines/metaot/sqeuclidean_eps0.01 --val experiments/torot/camera_ready

Meta OT (Amos, Cohen, Luise, Redko, ICML 2023), discrete setting: an MLP maps the two histograms (a, b) to the
entropic dual potential f; the second potential is the ε-c-transform g(f), and the plan is exp((f ⊕ g - C) / ε).
Training maximises the entropic dual objective on random pairs; no ground-truth plans are used.

Reused unchanged from the official code (third_party/, commit 7dcc9a4): the training loop ``Workspace.run`` of
train_discrete.py (Adam with gradient clipping, logging, checkpointing), the objective and plan construction
(``Workspace.dual_obj_loss / g_from_f / pred_transport``), the potential network (meta_ot.models.PotentialMLP)
and the hyperparameters of conf/train_discrete.yaml (MLP 3 x 1024, Adam 1e-3, clip 1.0, batch 128,
50,000 iterations, float64).
Supplied here for ToR-OT's setting: the input size (n = 30 bins instead of 784 pixels), the geometry (the
max-normalised 1D cost matrix of torot.data.cost instead of the 28 x 28 point cloud) and the pair sampler
(histograms drawn uniformly from the simplex, ToR-OT's training marginals), installed as ``meta_ot.data.RandomSampler``
so that the official ``run()`` uses it with ``data: random``.
Outputs in --out: latest.pkl + log.csv (official checkpoint and log), params.msgpack (the trained network, used by
predict.py), config.json, metrics.json (validation on sets_val/, used to choose ε).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party"))

import jax                                  # noqa: E402
import jax.numpy as jnp                     # noqa: E402

# Import-time compatibility with jax 0.4.30 for the unmodified official code (both names were removed from jax):
if not hasattr(jnp, "DeviceArray"):
    jnp.DeviceArray = jnp.ndarray           # used by ott-jax 0.2.9 isinstance checks
if "jax.config" not in sys.modules:         # train_discrete.py: `from jax.config import config`
    _cfg_module = types.ModuleType("jax.config")
    _cfg_module.config = jax.config
    sys.modules["jax.config"] = _cfg_module

import train_discrete                       # noqa: E402  (official; enables float64)
from meta_ot import data as meta_data       # noqa: E402
import flax.serialization                   # noqa: E402
from omegaconf import OmegaConf             # noqa: E402
from hydra.utils import instantiate         # noqa: E402
from ott.geometry.geometry import Geometry  # noqa: E402

from baselines.common import cost_matrix, find_sets, load_set, plan_summary, provenance, write_json  # noqa: E402

sys.excepthook = sys.__excepthook__         # the official module installs an interactive (pdb) excepthook
CONF = os.path.join(HERE, "third_party", "conf", "train_discrete.yaml")


def simplex_sampler(C: np.ndarray, eps: float):
    """A drop-in for meta_ot.data.RandomSampler: batches of histogram pairs uniform on the simplex, 1D geometry."""
    n = C.shape[0]

    class SimplexPairSampler:
        def __init__(self, batch_size: int = 128, debug: bool = False):
            self.batch_size = batch_size
            self.geom = Geometry(cost_matrix=jnp.asarray(C), epsilon=eps, relative_epsilon=False)

        def __call__(self, key):
            ka, kb = jax.random.split(key)
            alpha = jnp.ones(n)
            return meta_data.PairData(jax.random.dirichlet(ka, alpha, (self.batch_size,)),
                                      jax.random.dirichlet(kb, alpha, (self.batch_size,)))

    return SimplexPairSampler


class Workspace1D(train_discrete.Workspace):
    """The official Workspace with the input size set to n (its constructor hard-codes the datasets' sizes)."""

    def __init__(self, cfg, n: int, C: np.ndarray, eps: float):
        self.work_dir = os.getcwd()
        self.cfg, self.train_iter, self.output, self.n_output = cfg, 0, 0, n
        self.key = jax.random.PRNGKey(cfg.seed)
        self.potential_model = instantiate(cfg.potential_model, n_output=n)
        init_key, self.key = jax.random.split(self.key)
        self.params = self.potential_model.init(init_key, jnp.zeros(n), jnp.zeros(n))["params"]
        self.geom = Geometry(cost_matrix=jnp.asarray(C), epsilon=eps, relative_epsilon=False)

    def predict(self, a: np.ndarray, b: np.ndarray, batch: int = 1024) -> np.ndarray:
        """Plans exp((f + g(f) - C) / ε) for a batch of pairs (official pred_transport, vmapped and jitted)."""
        if getattr(self, "_predict_fn", None) is None:
            self._predict_fn = jax.jit(jax.vmap(self.pred_transport))
        out = [self._predict_fn(jnp.asarray(a[s:s + batch]), jnp.asarray(b[s:s + batch])) for s in range(0, len(a), batch)]
        return np.concatenate([np.asarray(jax.block_until_ready(o)) for o in out])

    @classmethod
    def load(cls, run: str) -> tuple["Workspace1D", dict]:
        """Rebuild a trained workspace from <run>/config.json and <run>/params.msgpack."""
        meta = json.load(open(f"{run}/config.json"))
        ws = cls(OmegaConf.create(meta["official_config"]), meta["n"], cost_matrix(meta["n"], meta["cost"], meta.get("dim", 1)),
                 meta["eps"])
        with open(f"{run}/params.msgpack", "rb") as fp:
            ws.params = flax.serialization.from_bytes(ws.params, fp.read())
        return ws, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cost", required=True)
    ap.add_argument("--eps", type=float, required=True, help="entropic regularisation (absolute; the cost max is 1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", required=True, help="results folder: validation set <root>/sets_val/val_simplex_<cost>.npz")
    ap.add_argument("--n", type=int, default=30, help="number of bins (2D: side^2, e.g. 36 for 6 x 6)")
    ap.add_argument("--dim", type=int, default=1, choices=[1, 2], help="2: bins on a sqrt(n) x sqrt(n) grid; validation set val_simplex2d_<cost>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-train-iter", type=int, default=None, help="default: official value (50,000)")
    args = ap.parse_args()

    val_path = find_sets(args.val, [f"val_simplex{'2d' if args.dim == 2 else ''}_{args.cost}"], folder="sets_val")[0]
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    cfg = OmegaConf.load(CONF)
    cfg.pop("hydra", None)
    cfg.data, cfg.seed = "random", args.seed
    if args.num_train_iter is not None:
        cfg.num_train_iter = args.num_train_iter
    C = cost_matrix(args.n, args.cost, args.dim)
    meta_data.RandomSampler = simplex_sampler(C, args.eps)       # picked up by the official run() for data: random

    write_json(f"{out}/config.json", {
        "method": "metaot", "cost": args.cost, "eps": args.eps, "n": args.n, "dim": args.dim, "seed": args.seed,
        "train_marginals": "uniform on the simplex (Dirichlet(1)), as ToR-OT's training data",
        "official_config": OmegaConf.to_container(cfg, resolve=True), "vendored_commit": "7dcc9a4c25b0a57626d3fa26028c6633ad19da50",
        "versions": {m: __import__(m).__version__ for m in ("jax", "jaxlib", "flax", "optax", "numpy")} | {"ott-jax": __import__("ott").__version__},
        "device": str(jax.devices()[0]), "provenance": provenance()})

    cwd = os.getcwd()
    os.chdir(out)                                                 # the official Workspace logs and saves to cwd
    try:
        ws = Workspace1D(cfg, args.n, C, args.eps)
        t0 = time.time()
        ws.run()
        train_s = time.time() - t0
    finally:
        os.chdir(cwd)

    with open(f"{out}/params.msgpack", "wb") as fp:
        fp.write(flax.serialization.to_bytes(ws.params))
    s = load_set(val_path)
    plans = ws.predict(s["a"], s["b"])
    n_params = int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(ws.params)))
    write_json(f"{out}/metrics.json", {"validation": plan_summary(plans, s), "train_seconds": train_s,
                                       "parameters": n_params, "iterations": int(cfg.num_train_iter)})
    print(f"done: validation paper MAPE {plan_summary(plans, s)['paper_mape_mean']:.4f}, {train_s:.0f} s", flush=True)


if __name__ == "__main__":
    main()
