"""Train UNOT on 1D histograms with the official code (run in env_unot, from the repository root).

    python -m baselines.unot.train --cost sqeuclidean --eps 0.01 \\
        --out experiments/baselines/unot/sqeuclidean_eps0.01 --val experiments/torot/camera_ready

UNOT (Geuter, Kornhardt, Tomasson, Laschos, ICML 2025): a predictor network maps (mu, nu) to the log Sinkhorn
scaling log v. Its targets come from a few Sinkhorn iterations warm-started at its own prediction, compared with
the Hilbert projective loss; the training measures come from a generator network trained adversarially to
maximise that loss, so no dataset is needed.

Reused unchanged from the official code (third_party/, commit 73b290b): the training loop
``src.train.train_mlp.Training`` (alternating predictor / generator updates, Sinkhorn targets ``src.ot.sinkhorn``,
Hilbert projective loss ``src.loss``), the predictor ``src.networks.mlp.Predictor``, the generator layers
``src.networks.generator.Generator`` and the default hyperparameters of scripts/main_mlp.py (AdamW lr 0.01, weight
decay 1e-4 predictor / 0 generator, exponential lr decay 0.9999 predictor / 1.0 generator, batch 64, 10,000
iterations, 5 Sinkhorn iterations per target, 3 hidden layers, latent prior of length 10, dust 1e-6, skip 1, float32).
UNOT's headline model is a Fourier neural operator for 2D grids of any resolution; for fixed-size inputs the
authors provide this MLP variant, which is the one that applies to 30-bin 1D histograms.

Supplied here for ToR-OT's setting:
* the cost matrix: the max-normalised 1D cost of torot.data.cost (instead of the 2D grid cost);
* Generator1D: the official generator's skip connection resizes a square latent image to the square measure image
  (torchvision Resize); in 1D it resizes the latent vector with linear interpolation. The rest of its forward pass
  is the official code path;
* the width of both networks: 1024 by default, the width of the MetaOT baseline; the official rule (4 x input size)
  gives 120 for 30 bins and is trained as well (--width 120); validation chooses (reproduce/run_learned_baselines.sh).
Outputs in --out: predictor_trained.pt (as left by the official loop), predictor.pt (after BatchNorm recalibration,
see recalibrate_batchnorm), generator.pt, losses.json, config.json, metrics.json (validation on sets_val/).
Runs trained before the recalibration step was added are finalised with ``python -m baselines.unot.recalibrate``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party"))

import wandb                                            # noqa: E402  (the official loop logs to wandb; disabled below)
from src.networks.generator import Generator            # noqa: E402
from src.networks.mlp import Predictor                  # noqa: E402
from src.train.train_mlp import Training                # noqa: E402

from baselines.common import cost_matrix, find_sets, load_set, plan_summary, provenance, write_json  # noqa: E402

DEFAULTS = {  # scripts/main_mlp.py defaults (official), except width (see module docstring)
    "lr": 0.01, "weight_decay_generator": 0.0, "weight_decay_predictor": 1e-4, "gamma_generator": 1.0,
    "gamma_predictor": 0.9999, "batch_size": 64, "nits": 10000, "nits_mini_loop_generator": 1,
    "nits_mini_loop_predictor": 1, "sinkhorn_max_iterations": 5, "num_layers": 3, "length_latent": 10,
    "dust_const": 1e-6, "skip_const": 1.0, "width": 1024}


def log_domain_targets(self, mu, nu):
    """Drop-in for the official ``Training.target_sinkhorn``: the same Sinkhorn iterations warm-started at the
    predictor's output (u = mu / (K v), v = nu / (K^T u), ``sinkhorn_max_iterations`` times), computed in the log domain
    so that small eps cannot underflow. Returns log v and the mask of finite rows, like the official method."""
    with torch.no_grad():
        log_v = self.predictor(mu, nu)                       # the official code exponentiates this as the initial v
        logK = -self.cost_matrix / self.sinkhorn_epsilon
        log_mu, log_nu = torch.log(mu), torch.log(nu)
        for _ in range(self.sinkhorn_max_iterations):
            log_u = log_mu - torch.logsumexp(logK[None] + log_v[:, None, :], dim=2)
            log_v = log_nu - torch.logsumexp(logK[None] + log_u[:, :, None], dim=1)
    return log_v, torch.isfinite(log_v).all(dim=1).to(self.device)


class Generator1D(Generator):
    """The official generator with a 1D skip connection (latent vector -> n bins by linear interpolation)."""

    def forward(self, x):
        x_0 = x.reshape(2, x.size(0), 1, self.dim_prior)                       # official: (2, B, L_prior, L_prior)
        resize = lambda t: nn.functional.interpolate(t, size=self.dim, mode="linear", align_corners=False)
        x_0 = torch.cat((resize(x_0[0]).reshape(x.size(0), self.dim), resize(x_0[1]).reshape(x.size(0), self.dim)), 1)
        for layer in self.layers:                                              # official from here on
            x = layer(x)
        x = x + self.skip_const * x_0
        x = nn.functional.relu(x)
        x_a, x_b = x[:, : self.dim], x[:, self.dim:]
        x_a = x_a / torch.unsqueeze(x_a.sum(dim=1), 1)
        x_b = x_b / torch.unsqueeze(x_b.sum(dim=1), 1)
        x_a, x_b = x_a + self.dust_const, x_b + self.dust_const
        x_a = x_a / torch.unsqueeze(x_a.sum(dim=1), 1)
        x_b = x_b / torch.unsqueeze(x_b.sum(dim=1), 1)
        return x_a, x_b


def plans_from_predictor(predictor: Predictor, a: np.ndarray, b: np.ndarray, C: np.ndarray, eps: float,
                         device: torch.device, batch: int = 4096, sinkhorn_steps: int = 0) -> np.ndarray:
    """P = diag(u) K diag(v) with v = exp(predictor(a, b)) and u = a / (K v): the plan UNOT hands to Sinkhorn as
    its initialisation (one half-step), computed in the log domain in float64 so that small eps cannot underflow."""
    predictor.eval()
    logK = -torch.as_tensor(C, dtype=torch.float64, device=device) / eps
    out = []
    with torch.no_grad():
        for s in range(0, len(a), batch):
            A = torch.as_tensor(a[s:s + batch], dtype=torch.float32, device=device)
            B = torch.as_tensor(b[s:s + batch], dtype=torch.float32, device=device)
            log_v = predictor(A, B).double()
            log_u = torch.log(A.double()) - torch.logsumexp(logK[None] + log_v[:, None, :], dim=2)
            for _ in range(sinkhorn_steps):                    # optional Sinkhorn refinement (UNOT's intended use)
                log_v = torch.log(B.double()) - torch.logsumexp(logK[None] + log_u[:, :, None], dim=1)
                log_u = torch.log(A.double()) - torch.logsumexp(logK[None] + log_v[:, None, :], dim=2)
            out.append(torch.exp(log_u[:, :, None] + logK[None] + log_v[:, None, :]).cpu().numpy())
    return np.concatenate(out)


def latent_size(hp: dict, dim: int) -> int:
    """Generator prior size: 1D adapter = length_latent; 2D = length_latent^2, the official image generator's prior."""
    return hp["length_latent"] ** 2 if dim == 2 else hp["length_latent"]


def build(n: int, hp: dict, device: torch.device, dim: int = 1) -> tuple[Generator, Predictor]:
    """1D: the official generator with a 1D skip connection (Generator1D). 2D: the official generator unchanged (square
    images of sqrt(n) x sqrt(n) bins, latent image of length_latent x length_latent), as in scripts/main_mlp.py."""
    G = Generator if dim == 2 else Generator1D
    generator = G(dim_prior=latent_size(hp, dim), dim=n, dim_hidden=hp["width"], num_layers=hp["num_layers"],
                  dust_const=hp["dust_const"], skip_const=hp["skip_const"]).to(device)
    predictor = Predictor(n, hp["width"], num_layers=hp["num_layers"]).to(device)
    return generator, predictor


def load(run: str, device: torch.device) -> tuple[Predictor, dict]:
    meta = json.load(open(f"{run}/config.json"))
    _, predictor = build(meta["n"], meta["hyperparameters"], device, meta.get("dim", 1))
    predictor.load_state_dict(torch.load(f"{run}/predictor.pt", map_location=device, weights_only=True))
    return predictor.eval(), meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cost", required=True)
    ap.add_argument("--eps", type=float, required=True, help="entropic regularisation (absolute; the cost max is 1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", required=True, help="results folder: validation set <root>/sets_val/val_simplex_<cost>.npz")
    ap.add_argument("--n", type=int, default=30, help="number of bins (2D: side^2, e.g. 36 for 6 x 6)")
    ap.add_argument("--dim", type=int, default=1, choices=[1, 2], help="2: bins on a sqrt(n) x sqrt(n) grid, official image generator")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nits", type=int, default=None, help="default: official value (10,000)")
    ap.add_argument("--width", type=int, default=None, help="hidden width of both networks (default 1024; official rule: 4 x n)")
    ap.add_argument("--log-targets", action="store_true",
                    help="compute the Sinkhorn targets in the log domain (identical iterates; the official float32 scaling-domain "
                         "Sinkhorn underflows for small eps on this cost scale)")
    args = ap.parse_args()

    hp = dict(DEFAULTS, **({"nits": args.nits} if args.nits else {}), **({"width": args.width} if args.width else {}))
    val_path = find_sets(args.val, [f"val_simplex{'2d' if args.dim == 2 else ''}_{args.cost}"], folder="sets_val")[0]
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    C = cost_matrix(args.n, args.cost, args.dim)
    generator, predictor = build(args.n, hp, device, args.dim)
    write_json(f"{args.out}/config.json", {
        "method": "unot", "cost": args.cost, "eps": args.eps, "n": args.n, "dim": args.dim, "seed": args.seed, "hyperparameters": hp,
        "train_marginals": "UNOT's adversarial generator (no dataset), as in the official method",
        "sinkhorn_targets": "log domain (identical iterates)" if args.log_targets else "official (float32, scaling domain)",
        "vendored_commit": "73b290ba8c0709fa7dd4992eeda88171628d6bb2",
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "wandb": wandb.__version__},
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu", "provenance": provenance()})

    wandb.init(mode="disabled")
    trainer = Training(
        generator, predictor, lr=hp["lr"], weight_decay_generator=hp["weight_decay_generator"],
        weight_decay_predictor=hp["weight_decay_predictor"], multiplicative_factor_generator=hp["gamma_generator"],
        multiplicative_factor_predictor=hp["gamma_predictor"], numbr_training_iterations=hp["nits"],
        numbr_mini_loop_predictor=hp["nits_mini_loop_predictor"], numbr_mini_loop_generator=hp["nits_mini_loop_generator"],
        sinkhorn_max_iterations=hp["sinkhorn_max_iterations"], sinkhorn_epsilon=args.eps,
        cost_matrix=torch.as_tensor(C, dtype=torch.float32, device=device), batch_size=hp["batch_size"],
        dim_prior=latent_size(hp, args.dim), test_data=None, device=device)
    if args.log_targets:
        trainer.target_sinkhorn = types.MethodType(log_domain_targets, trainer)
    generator.train(); predictor.train()
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0

    torch.save(predictor.state_dict(), f"{args.out}/predictor_trained.pt")       # as left by the official loop
    torch.save(generator.state_dict(), f"{args.out}/generator.pt")
    write_json(f"{args.out}/losses.json", trainer.train_losses)
    finalize(args.out, predictor, generator, val_path, device, {"train_seconds": train_s})


def recalibrate_batchnorm(predictor: Predictor, generator: Generator1D, latent: int, device: torch.device,
                          batches: int = 200, batch_size: int = 64, seed: int = 1) -> dict:
    """Re-estimate the predictor's BatchNorm running statistics on finite samples of the trained generator.

    The official loop runs every generated batch through the predictor in train mode (``Training.target_sinkhorn``),
    which updates the running statistics. With 30-bin histograms the generator occasionally outputs an all-zero
    measure (ReLU), which its normalisation turns into NaN; one such batch makes the running statistics NaN for
    good, so the predictor outputs NaN in eval mode. Training itself is unaffected (the loss only uses samples
    whose Sinkhorn target is finite). The weights are not changed: only the running averages are recomputed
    (cumulative average over ``batches`` batches of the method's own training distribution, the generator).
    """
    torch.manual_seed(seed)
    bns = [m for m in predictor.modules() if isinstance(m, nn.BatchNorm1d)]
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats(); m.momentum = None
    predictor.train(); generator.train()
    used = dropped = 0
    with torch.no_grad():
        for _ in range(batches):
            mu, nu = generator(torch.randn(batch_size, 2 * latent, device=device))
            ok = torch.isfinite(mu).all(1) & torch.isfinite(nu).all(1)
            dropped += int((~ok).sum())
            if ok.sum() > 1:
                predictor(mu[ok], nu[ok]); used += int(ok.sum())
    for m, mom in zip(bns, momenta):
        m.momentum = mom
    predictor.eval()
    return {"batchnorm_recalibrated_on": f"{used} finite generator samples ({dropped} non-finite dropped)"}


def finalize(run: str, predictor: Predictor, generator: Generator1D, val_path: str, device: torch.device, extra: dict):
    """BatchNorm recalibration, final checkpoint (predictor.pt) and validation metrics (used to choose eps)."""
    meta = json.load(open(f"{run}/config.json"))
    hp, losses = meta["hyperparameters"], json.load(open(f"{run}/losses.json"))["predictor"]
    info = recalibrate_batchnorm(predictor, generator, latent_size(hp, meta.get("dim", 1)), device)
    torch.save(predictor.state_dict(), f"{run}/predictor.pt")
    s = load_set(val_path)
    plans = plans_from_predictor(predictor, s["a"], s["b"], cost_matrix(meta["n"], meta["cost"], meta.get("dim", 1)), meta["eps"], device)
    lf = np.asarray(losses, dtype=float)
    metrics = {"validation": plan_summary(plans, s), **extra, "parameters": int(sum(p.numel() for p in predictor.parameters())),
               "iterations": hp["nits"], "predictor_updates": len(lf), "non_finite_predictor_losses": int((~np.isfinite(lf)).sum()),
               "final_predictor_loss_mean_last_100": float(np.mean(lf[-100:])) if len(lf) else float("nan"), **info}
    write_json(f"{run}/metrics.json", metrics)
    print(f"done: validation paper MAPE {metrics['validation']['paper_mape_mean']:.4f}, {len(lf)} predictor updates "
          f"({metrics['non_finite_predictor_losses']} non-finite), {info['batchnorm_recalibrated_on']}", flush=True)


if __name__ == "__main__":
    main()
