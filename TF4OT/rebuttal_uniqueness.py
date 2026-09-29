"""
Rebuttal experiment (Q1, non-uniqueness of discrete OT solutions):
How often does the exact LP admit an alternative optimum, per cost metric?

Two tests are run per instance:

  (A) Reduced-cost criterion (the test named in the rebuttal text).
      From POT's optimal duals (u, v), the reduced cost is R_ij = C_ij - u_i - v_j.
      At optimality R >= 0 and R = 0 on the support. If some entry OUTSIDE the
      support has R_ij = 0, the optimal vertex is not the unique optimum
      (a pivot into (i,j) keeps the objective). This is necessary; it is also
      sufficient when the primal is non-degenerate (|support| = p + q - 1).

  (B) Exact LP certificate (valid with or without degeneracy).
      By complementary slackness every optimal plan is supported on the
      zero-reduced-cost set Z = {(i,j) : R_ij = 0}, and conversely any pi >= 0
      supported on Z with the right marginals has cost
          <C, pi> = sum pi_ij (u_i + v_j) = <u,a> + <v,b> = v*,
      hence is optimal. So the optimum is non-unique iff
          max sum_{(i,j) in Z \ supp(pi*)} pi_ij
          s.t. pi >= 0 supported on Z, pi 1 = a, pi^T 1 = b
      is > 0. This needs no cost constraint, so it does not depend on a
      numerically delicate objective tolerance; it depends only on the
      classification of reduced costs as zero, which is unambiguous (the
      smallest strictly positive reduced cost is many orders of magnitude
      above the floating-point zeros -- reported in the output table).
      A value of 0 means every optimal plan is supported in supp(pi*), which is
      a forest, on which the marginals determine the flow uniquely. This is what
      separates a genuine alternative optimum from a merely degenerate pivot,
      which test (A) alone cannot do.

Zero-mass bins are removed first (a rows / b columns with no mass carry no flow
and have arbitrary duals), so all tests run on the reduced p x q problem.

Corpora: 1D in-distribution (test set), 1D OOD (named distribution pairs),
2D in-distribution (test set), 2D OOD (named 2D distribution pairs).

Run:
  HF_DATASETS_OFFLINE=1 PYTHONPATH=. \
    /home/ivan/anaconda3/envs/env_TF4OT/bin/python TF4OT/rebuttal_uniqueness.py
"""

import os
import json
import numpy as np
import torch
import ot
import scipy.sparse as sp
from scipy.optimize import linprog
from tabulate import tabulate
from sklearn import datasets as skdatasets

from TF4OT import utils
from TF4OT.utils import my_dtype_np

N_ID          = os.environ.get("N_ID", "300")        # instances per split, or "all"
N_WORKERS     = int(os.environ.get("N_WORKERS", max(1, (os.cpu_count() or 2) - 4)))
N_BINS_1D     = 30
N_BINS_2D     = 36                                    # 6 x 6 grid
N_SAMPLES_1D  = 1_000_000                             # as in eval_all_1d_distribution_pairs
N_SAMPLES_2D  = 100_000                               # as in eval_all_2d_distribution_pairs
HF_CACHE      = "/data/ivan/datasets/hfcache"
DS_1D         = "ivanjaenm/otdata_100K_rowwise_costtype_sqeuclidean_bins30"
DS_2D         = "ivanjaenm/otdata_100K_rowwise_costtype_sqeuclidean_dim2_bins36"
OUT_DIR       = "experiments/cost_fixed"
SEED          = 0

TOL_MASS   = 1e-12   # a_i / b_j treated as zero below this
TOL_PLAN   = 1e-12   # plan entry treated as zero below this
TOL_RC     = 1e-9    # reduced cost treated as zero below this (C is normalised to max 1)
TOL_ALT    = 1e-9    # LP: mass placeable outside the support to call the optimum non-unique


# ----------------------------------------------------------------------------- costs
def cost_matrix(metric, dim, n_bins):
    """
    Cost matrix as built by run_datagenerator (p=3 for minkowski).

    minkowski is computed directly with cbrt rather than through scipy's
    pow-based cdist: the latter carries ~1e-7 relative error, which is large
    enough to split cost ties that are exactly zero in exact arithmetic and
    would therefore corrupt the uniqueness test. (In 1D this makes
    minkowski-p3 bitwise identical to |i-j|, as it is mathematically.)
    """
    # float64 throughout: the support points are exact integers (1D) / grid
    # values (2D); the pipeline's float32 storage would otherwise inject ~1e-7
    # rounding that splits exact cost ties.
    if dim == 1:
        xs = np.arange(n_bins, dtype=np.float64)[:, None]
    else:
        s = int(np.sqrt(n_bins))
        rng = [[-2.5, 2.5], [-2.5, 2.5]]
        dx = (rng[0][1] - rng[0][0]) / s
        dy = (rng[1][1] - rng[1][0]) / s
        xc = np.linspace(rng[0][0] + dx / 2, rng[0][1] - dx / 2, s)
        yc = np.linspace(rng[1][0] + dy / 2, rng[1][1] - dy / 2, s)
        XX, YY = np.meshgrid(xc, yc, indexing="ij")
        xs = np.stack([XX.flatten(), YY.flatten()], axis=1)
    if metric == "minkowski":
        d = np.abs(xs[:, None, :] - xs[None, :, :])
        C = np.cbrt((d ** 3).sum(axis=-1))
    else:
        C = np.asarray(ot.dist(xs, xs, metric=metric), dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    return C / C.max()


# ----------------------------------------------------------------------------- test
def analyse_instance(C, a, b):
    """Run tests (A) and (B) on one instance. Returns a dict of diagnostics."""
    I = np.where(a > TOL_MASS)[0]
    J = np.where(b > TOL_MASS)[0]
    Cs = np.ascontiguousarray(C[np.ix_(I, J)])
    asub = a[I] / a[I].sum()
    bsub = b[J] / b[J].sum()
    p, q = len(I), len(J)

    res = ot.solve(Cs, a=asub, b=bsub)
    plan = np.asarray(res.plan, dtype=np.float64)
    u, v = res.potentials
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    value = float((plan * Cs).sum())

    supp = plan > TOL_PLAN
    nnz = int(supp.sum())
    degenerate = nnz != (p + q - 1)

    # (A) reduced-cost criterion
    R = Cs - u[:, None] - v[None, :]
    Z = R <= TOL_RC                                # zero-reduced-cost set
    off = ~supp
    n_zero_off = int((Z & off).sum())
    rc_alt = n_zero_off > 0
    # tolerance headroom: largest |R| on the support vs smallest R outside Z
    max_rc_on_supp = float(np.abs(R[supp]).max()) if supp.any() else 0.0
    pos = R[~Z]
    min_pos_rc = float(pos.min()) if pos.size else np.nan

    # (B) exact LP certificate, restricted to Z (no cost constraint needed)
    zr, zc = np.nonzero(Z)
    k = len(zr)
    t = np.arange(k)
    A_eq = sp.vstack([
        sp.coo_matrix((np.ones(k), (zr, t)), shape=(p, k)),
        sp.coo_matrix((np.ones(k), (zc, t)), shape=(q, k)),
    ], format="csr")
    b_eq = np.concatenate([asub, bsub])
    obj = -(off[zr, zc]).astype(np.float64)        # maximise mass outside supp(pi*)
    lp = linprog(obj, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    mass_outside = float(-lp.fun) if lp.success else np.nan
    lp_alt = bool(lp.success and mass_outside > TOL_ALT)

    # Any optimal pi' attaining `mass_outside` differs from pi* by at least
    # 2 * mass_outside in L1: it places that mass off supp(pi*) and must remove
    # the same amount from within it. This lower-bounds the L1 diameter of the
    # optimal face, i.e. how much plan-level error is attributable to the choice
    # of optimum alone rather than to prediction error.
    l1_gap = 2.0 * max(mass_outside, 0.0)

    return dict(p=p, q=q, nnz=nnz, degenerate=degenerate,
                rc_alt=rc_alt, n_zero_off=n_zero_off, min_pos_rc=min_pos_rc,
                max_rc_on_supp=max_rc_on_supp,
                lp_alt=lp_alt, mass_outside=mass_outside, l1_gap=l1_gap,
                lp_ok=bool(lp.success), value=value)


_WORKER_C = None


def _pool_init(C):
    global _WORKER_C
    _WORKER_C = C
    # each worker runs single-threaded BLAS; parallelism comes from the pool
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _pool_task(ab):
    a, b = ab
    return analyse_instance(_WORKER_C, a, b)


def analyse_corpus(C, margs, n_workers=N_WORKERS):
    """analyse_instance over all (a, b) pairs, in parallel across processes."""
    if n_workers <= 1 or len(margs) < 64:
        return [analyse_instance(C, a, b) for a, b in margs]
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers, initializer=_pool_init, initargs=(C,)) as pool:
        return pool.map(_pool_task, margs, chunksize=max(1, len(margs) // (n_workers * 8)))


# ----------------------------------------------------------------------------- corpora
def marginals_from_testset(ds_name, n_bins, n_instances, split="test"):
    """
    Read (a, b) pairs from an HF split (row-wise: n_bins rows per instance).
    Only the first row of each block is read -- the marginals are identical
    across a block's rows. n_instances may be "all".
    """
    from datasets import load_dataset
    ds = load_dataset(ds_name, cache_dir=HF_CACHE)[split]
    total = len(ds) // n_bins
    n_instances = total if n_instances == "all" else min(int(n_instances), total)
    rows = ds.select(range(0, n_instances * n_bins, n_bins))["input_total"]
    out = []
    for vec in rows:
        vec = np.asarray(vec, dtype=np.float64)
        a = vec[:n_bins].copy()
        b = vec[n_bins:2 * n_bins].copy()
        out.append((a / a.sum(), b / b.sum()))
    return out


def samplers_1d():
    """Copied verbatim from eval_all_1d_distribution_pairs (run_inference.py)."""
    class TorchSampler:
        def __init__(self, fn, name=None):
            self.fn = fn
            self.name = name or "TorchSampler"
        def sample(self, n):
            return self.fn(n).float().view(-1, 1)

    def zscore(x, eps=1e-8):
        return (x - x.mean()) / (x.std() + eps)

    def Beta(a, b, name):
        m = torch.distributions.Beta(torch.tensor([a]), torch.tensor([b]))
        return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)), name=name)

    def Laplace():
        m = torch.distributions.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)), name="Laplace")

    def Uniform(low=-2.0, high=2.0):
        return TorchSampler(lambda n: zscore(torch.rand(n) * (high - low) + low), name="Uniform")

    def StudentT(df=5):
        m = torch.distributions.StudentT(torch.tensor([float(df)]))
        return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)), name="StudentT")

    def GaussianShifted(mu=1.0, sigma=1.0):
        return TorchSampler(lambda n: zscore(torch.randn(n) * sigma + mu), name="GaussianShifted")

    def LogNormal(mu=0.0, sigma=0.6):
        m = torch.distributions.LogNormal(torch.tensor([mu]), torch.tensor([sigma]))
        return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)), name="LogNormal")

    def CauchyTrunc(clip=5.0):
        m = torch.distributions.Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
        return TorchSampler(lambda n: zscore(torch.clamp(m.sample((n,)).view(-1), -clip, clip)),
                            name="CauchyTrunc")

    def Triangular(left=-3.0, mode=0.0, right=3.0):
        def _f(n):
            u = torch.rand(n)
            c = (mode - left) / (right - left)
            x = torch.zeros_like(u)
            mask = u < c
            if mask.any():
                x[mask] = left + torch.sqrt(u[mask] * (right - left) * (mode - left))
            if (~mask).any():
                x[~mask] = right - torch.sqrt((1 - u[~mask]) * (right - left) * (right - mode))
            return zscore(x)
        return TorchSampler(_f, name="Triangular")

    def Exponential(scale=1.0):
        m = torch.distributions.Exponential(torch.tensor([1.0 / scale]))
        return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)), name="Exponential")

    def Weibull(a=2.0, scale=1.0):
        m = torch.distributions.Weibull(torch.tensor([scale]), torch.tensor([a]))
        return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)), name="Weibull")

    def Gamma(shape=2.0, scale=1.0):
        m = torch.distributions.Gamma(torch.tensor([shape]), torch.tensor([1.0 / scale]))
        return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)), name="Gamma")

    return [Beta(2.0, 5.0, "BetaLeft"), Beta(5.0, 2.0, "BetaRight"), Laplace(), Uniform(),
            StudentT(5), GaussianShifted(1.0), LogNormal(sigma=0.6), CauchyTrunc(5.0),
            Triangular(-3, 0, 3), Exponential(1.0), Weibull(2.0), Gamma(2.0)]


def samplers_2d():
    """Copied verbatim from eval_all_2d_distribution_pairs (run_inference_2d.py)."""
    class TorchSampler:
        def __init__(self, fn, name=None):
            self.fn = fn
            self.name = name or "TorchSampler"
        def sample(self, n):
            return self.fn(n).float()

    def standardize(x):
        x = x - x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        std[std == 0] = 1.0
        return x / std

    def SwissRoll(noise=0.8):
        def _f(n):
            data, _ = skdatasets.make_swiss_roll(n_samples=n, noise=noise)
            return standardize(torch.tensor(data[:, [0, 2]], dtype=torch.float32))
        return TorchSampler(_f, name="SwissRoll")

    def Mix8Gaussians(std=0.5, r=12.0):
        centers = torch.tensor([(1, 0), (-1, 0), (0, 1), (0, -1),
                                (1 / np.sqrt(2), 1 / np.sqrt(2)), (1 / np.sqrt(2), -1 / np.sqrt(2)),
                                (-1 / np.sqrt(2), 1 / np.sqrt(2)), (-1 / np.sqrt(2), -1 / np.sqrt(2))],
                               dtype=torch.float32) * r
        def _f(n):
            idx = torch.randint(0, len(centers), (n,))
            return standardize(centers[idx] + torch.randn(n, 2) * std)
        return TorchSampler(_f, name="Mix8Gaussians")

    def StandardNormal():
        return TorchSampler(lambda n: torch.randn(n, 2), name="StandardNormal")

    def Moons(noise=0.1):
        def _f(n):
            data, _ = skdatasets.make_moons(n_samples=n, noise=noise)
            return standardize(torch.tensor(data, dtype=torch.float32))
        return TorchSampler(_f, name="Moons")

    def Circles(noise=0.05, factor=0.5):
        def _f(n):
            data, _ = skdatasets.make_circles(n_samples=n, noise=noise, factor=factor)
            return standardize(torch.tensor(data, dtype=torch.float32))
        return TorchSampler(_f, name="Circles")

    def SCurve(noise=0.1):
        def _f(n):
            data, _ = skdatasets.make_s_curve(n_samples=n, noise=noise)
            return standardize(torch.tensor(data[:, [0, 2]], dtype=torch.float32))
        return TorchSampler(_f, name="SCurve")

    def UniformSquare(low=-2.0, high=2.0):
        return TorchSampler(lambda n: standardize(torch.rand(n, 2) * (high - low) + low),
                            name="UniformSquare")

    return [SwissRoll(), Mix8Gaussians(), StandardNormal(), Moons(),
            Circles(), SCurve(), UniformSquare()]


def marginals_ood_1d():
    ds = samplers_1d()
    out = []
    for A in ds:
        for B in ds:
            src = A.sample(N_SAMPLES_1D)
            tgt = B.sample(N_SAMPLES_1D)
            a, b, _ = utils.hist_from_samples(src, tgt, bins=N_BINS_1D, density=False)
            out.append((np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)))
    return out


def marginals_ood_2d():
    ds = samplers_2d()
    s = int(np.sqrt(N_BINS_2D))
    rng = [[-2.5, 2.5], [-2.5, 2.5]]
    out = []
    for A in ds:
        for B in ds:
            src = A.sample(N_SAMPLES_2D).cpu().numpy()
            tgt = B.sample(N_SAMPLES_2D).cpu().numpy()
            hx, _, _ = np.histogram2d(src[:, 0], src[:, 1], bins=[s, s], range=rng)
            hy, _, _ = np.histogram2d(tgt[:, 0], tgt[:, 1], bins=[s, s], range=rng)
            out.append((hx.flatten() / hx.sum(), hy.flatten() / hy.sum()))
    return out


# ----------------------------------------------------------------------------- main
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Building corpora ...", flush=True)
    corpora = {
        "1D train":    (1, N_BINS_1D, marginals_from_testset(DS_1D, N_BINS_1D, N_ID, split="train")),
        "1D test":     (1, N_BINS_1D, marginals_from_testset(DS_1D, N_BINS_1D, N_ID, split="test")),
        "1D OOD":      (1, N_BINS_1D, marginals_ood_1d()),
        "2D train":    (2, N_BINS_2D, marginals_from_testset(DS_2D, N_BINS_2D, N_ID, split="train")),
        "2D test":     (2, N_BINS_2D, marginals_from_testset(DS_2D, N_BINS_2D, N_ID, split="test")),
        "2D OOD":      (2, N_BINS_2D, marginals_ood_2d()),
    }
    for name, (_, _, m) in corpora.items():
        print(f"  {name}: {len(m)} instances", flush=True)

    # which cost matrices coincide in each geometry
    print("\nCost-matrix coincidences:")
    for dim, nb in [(1, N_BINS_1D), (2, N_BINS_2D)]:
        Cs = {m: cost_matrix(m, dim, nb) for m in utils.metrics}
        groups = []
        for m in utils.metrics:
            placed = False
            for g in groups:
                if np.abs(Cs[m] - Cs[g[0]]).max() < 1e-12:
                    g.append(m); placed = True; break
            if not placed:
                groups.append([m])
        print(f"  {dim}D ({nb} bins): " + " | ".join("=".join(g) for g in groups))

    rows = []
    raw = {}
    for corpus_name, (dim, nb, margs) in corpora.items():
        for metric in utils.metrics:
            C = cost_matrix(metric, dim, nb)
            recs = analyse_corpus(C, margs)
            n = len(recs)
            lp_alt   = np.array([r["lp_alt"] for r in recs])
            rc_alt   = np.array([r["rc_alt"] for r in recs])
            degen    = np.array([r["degenerate"] for r in recs])
            nzoff    = np.array([r["n_zero_off"] for r in recs])
            mass     = np.array([r["mass_outside"] for r in recs])
            l1       = np.array([r["l1_gap"] for r in recs])
            minrc    = np.array([r["min_pos_rc"] for r in recs])
            maxrcs   = np.array([r["max_rc_on_supp"] for r in recs])
            nnz      = np.array([r["nnz"] for r in recs])
            assert all(r["lp_ok"] for r in recs), "an LP certificate failed to solve"

            rows.append([corpus_name, metric, n,
                         f"{100 * lp_alt.mean():.1f}",
                         f"{100 * rc_alt.mean():.1f}",
                         f"{np.median(mass[lp_alt]):.3f}" if lp_alt.any() else "-",
                         f"{np.median(l1[lp_alt]):.3f}" if lp_alt.any() else "-",
                         f"{100 * degen.mean():.1f}",
                         f"{nnz.mean():.1f}",
                         f"{np.nanmin(minrc):.1e}",
                         f"{maxrcs.max():.1e}"])
            raw[f"{corpus_name}|{metric}"] = dict(
                n=n, pct_lp=100 * float(lp_alt.mean()), pct_rc=100 * float(rc_alt.mean()),
                agreement=100 * float((lp_alt == rc_alt).mean()),
                pct_degenerate=100 * float(degen.mean()), mean_nnz=float(nnz.mean()),
                median_mass_outside=float(np.median(mass[lp_alt])) if lp_alt.any() else 0.0,
                median_l1_gap=float(np.median(l1[lp_alt])) if lp_alt.any() else 0.0,
                max_l1_gap=float(l1.max()),
                mean_zero_rc_offsupport=float(nzoff[lp_alt].mean()) if lp_alt.any() else 0.0,
                min_positive_reduced_cost=float(np.nanmin(minrc)),
                max_reduced_cost_on_support=float(maxrcs.max()))
            print(f"  [{corpus_name:11s} {metric:12s}] non-unique: LP {100*lp_alt.mean():5.1f}%  "
                  f"RC {100*rc_alt.mean():5.1f}%  degenerate {100*degen.mean():4.1f}%", flush=True)

    headers = ["Corpus", "Cost", "N", "Non-unique % (LP)", "Non-unique % (reduced cost)",
               "Median movable mass", "Median L1 gap", "Primal degenerate %",
               "Mean nnz", "Min positive RC", "Max RC on support"]
    print("\n" + tabulate(rows, headers=headers, tablefmt="github"))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "rebuttal_uniqueness_results.json"), "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\nSaved -> {os.path.join(OUT_DIR, 'rebuttal_uniqueness_results.json')}")


if __name__ == "__main__":
    main()
