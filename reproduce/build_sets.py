"""Build the shared evaluation sets (n = m = 30) on which every method is run.

    python reproduce/build_sets.py experiments/torot/camera_ready [costs...] [--val-only] [--n N]   (env_torot, repo root)
    (--n 100: the same construction at n = 100 bins, written to its own root, e.g. experiments/torot/camera_ready_n100)

Sets (n = m = 30; costs sqeuclidean and euclidean by default, the other costs with `build_sets.py <dir> <cost>`; the
marginals are identical for every cost):

* ood_paper_<cost>  : the OOD set: 12 named distributions, all 144 ordered pairs. Each distribution is sampled
                      1,000,000 times (torch, seed 0), standardized, and each pair is binned on shared edges over the
                      pooled 0.5-99.5 percentile range (paper_families, hist_from_samples).
* id_simplex_<cost> : 1,000 random instances with the training marginals of the corpus (uniform on the simplex).

* sets_val/val_simplex_<cost> : 500 random instances with the training marginals of the corpus, for model selection of
                      the learned baselines (e.g. their entropic regularisation); never used for reporting.

ID seeds start at 2e9 (id_simplex) and 3e9 (validation): disjoint from each other and from the
training corpora (seeds < 1.1e7). Bins: bin centres in [0, 1].
"""
import glob, json, os, sys
import numpy as np, ot, torch
from torot.data.cost import cost_matrix
from torot.data.generate import GenConfig, make_random_instance
from torot.data.supports import grid_1d

OUT, N_ID, N_VAL = sys.argv[1], 1000, 500
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 30   # support size (e.g. --n 100, separate <root>)
VAL_ONLY = "--val-only" in sys.argv                              # only (re)build sets_val/ for the given costs
COSTS = tuple(a for i, a in enumerate(sys.argv[2:], 2) if not a.startswith("--") and sys.argv[i - 1] != "--n") \
    or ("sqeuclidean", "euclidean")


# ----------------------------------------------------------------- the samplers of the 12 OOD distributions
class TorchSampler:
    def __init__(self, fn, name):
        self.fn, self.name = fn, name

def zscore(x, eps=1e-8):
    return (x - x.mean()) / (x.std() + eps)

def BetaLeft(a=2.0, b=5.0):
    m = torch.distributions.Beta(torch.tensor([a]), torch.tensor([b])); return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)), "BetaLeft")
def BetaRight(a=5.0, b=2.0):
    m = torch.distributions.Beta(torch.tensor([a]), torch.tensor([b])); return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)), "BetaRight")
def LogNormal(mu=0.0, sigma=0.5):
    m = torch.distributions.LogNormal(torch.tensor([mu]), torch.tensor([sigma])); return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)), "LogNormal")
def Laplace(loc=0.0, scale=1.0):
    m = torch.distributions.Laplace(torch.tensor([loc]), torch.tensor([scale])); return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)), "Laplace")
def StudentT(df=3):
    m = torch.distributions.StudentT(torch.tensor([float(df)])); return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)), "StudentT")
def CauchyTrunc(x0=0.0, gamma=1.0, clip=6.0):
    m = torch.distributions.Cauchy(torch.tensor([x0]), torch.tensor([gamma]))
    return TorchSampler(lambda n: zscore(torch.clamp(m.sample((n,)).view(-1), -clip, clip)), "CauchyTrunc")
def Triangular(left=-2.0, mode=0.0, right=2.0):
    def _f(n):
        u = torch.rand(n); c = (mode - left) / (right - left); x = torch.zeros_like(u); mask = u < c
        if mask.any(): x[mask] = left + torch.sqrt(u[mask] * (right - left) * (mode - left))
        if (~mask).any(): x[~mask] = right - torch.sqrt((1 - u[~mask]) * (right - left) * (right - mode))
        return zscore(x)
    return TorchSampler(_f, "Triangular")
def UniformLowHigh(low=-2.0, high=2.0):
    return TorchSampler(lambda n: zscore(torch.rand(n) * (high - low) + low), "Uniform")
def GaussianShifted(mu=1.5, sigma=1.0):
    return TorchSampler(lambda n: zscore(torch.randn(n) * sigma + mu), "GaussianShifted")
def Exponential(scale=1.0):
    m = torch.distributions.Exponential(torch.tensor([1.0 / scale])); return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)), "Exponential")
def Weibull(a=1.5, scale=1.0):
    m = torch.distributions.Weibull(torch.tensor([scale]), torch.tensor([a])); return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)), "Weibull")
def Gamma(shape=2.0, scale=1.0):
    m = torch.distributions.Gamma(torch.tensor([shape]), torch.tensor([1.0 / scale])); return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)), "Gamma")

def paper_families():
    """The 12 OOD distributions, in the order of the set and with their parameters."""
    return [BetaLeft(), BetaRight(), Laplace(), UniformLowHigh(), StudentT(df=5), GaussianShifted(mu=1.0),
            LogNormal(sigma=0.6), CauchyTrunc(clip=5.0), Triangular(-3, 0, 3), Exponential(scale=1.0), Weibull(a=2.0), Gamma(2.0)]

def hist_from_samples(X, Y, bins, trim_percentiles=(0.5, 99.5)):
    """Histograms of two samples on shared bin edges over the pooled 0.5-99.5 percentile range."""
    X, Y = X.numpy().reshape(-1), Y.numpy().reshape(-1)
    pooled = np.concatenate([X, Y])
    edges = np.linspace(np.percentile(pooled, trim_percentiles[0]), np.percentile(pooled, trim_percentiles[1]), bins + 1)
    cx, _ = np.histogram(X, bins=edges); cy, _ = np.histogram(Y, bins=edges)
    return cx / cx.sum(), cy / cy.sum()


# ----------------------------------------------------------------- helpers
coords = grid_1d(N)

def solve(a, b, cost):
    C = cost_matrix(coords, coords, cost)
    P = ot.emd(a, b, C, numItermax=1_000_000)
    P[P < 1e-12] = 0.0
    return P, float((P * C).sum())

def save(name, cost, a, b, src, tgt, seeds, description, folder="sets"):
    plans, costs = zip(*[solve(x, y, cost) for x, y in zip(a, b)])
    np.savez(f"{OUT}/{folder}/{name}.npz", a=np.stack(a), b=np.stack(b), plan=np.stack(plans), ot_cost=np.array(costs),
             src=np.array(src), tgt=np.array(tgt), seed=np.array(seeds), cost=cost, n=N)
    manifest[name] = {"cost": cost, "instances": len(a), "description": description}
    print(f"{name:<28} {len(a):>5} instances", flush=True)

manifest = json.load(open(f"{OUT}/sets/manifest.json")) if os.path.exists(f"{OUT}/sets/manifest.json") else {}
torch.manual_seed(0); np.random.seed(0)
existing = sorted(glob.glob(f"{OUT}/sets/ood_paper_*.npz"))
if existing:   # reuse the histograms already built, so every cost is evaluated on identical OOD marginals
    z = np.load(existing[0])  # (re-sampling 1e6 points per family can move a few samples across bin edges)
    pairs = [(str(s), str(t), a, b) for s, t, a, b in zip(z["src"], z["tgt"], z["a"], z["b"])]
else:
    fams = paper_families()
    pairs = [(X.name, Y.name, *hist_from_samples(X.fn(1_000_000), Y.fn(1_000_000), N)) for X in fams for Y in fams]
for cost in COSTS:
    if not VAL_ONLY:
        save(f"ood_paper_{cost}", cost, [p[2] for p in pairs], [p[3] for p in pairs], [p[0] for p in pairs], [p[1] for p in pairs],
             [-1] * len(pairs), "OOD set (12 named distributions x 144 pairs), torch seed 0")
        marg, base = "simplex", 2 * 10**9
        cfg, recs, seed = GenConfig(n_min=N, n_max=N, cost_types=[cost], marginals=marg), [], base
        while len(recs) < N_ID:
            r = make_random_instance(seed, cfg); seed += 1
            if r is not None: recs.append(r)
        save(f"id_simplex_{cost}", cost, [np.asarray(r["a"]) for r in recs], [np.asarray(r["b"]) for r in recs],
             [marg] * N_ID, [marg] * N_ID, [r["seed"] for r in recs], f"{N_ID} random instances, {marg} marginals, seeds from {base}")
    os.makedirs(f"{OUT}/sets_val", exist_ok=True)
    cfg, recs, seed = GenConfig(n_min=N, n_max=N, cost_types=[cost], marginals="simplex"), [], 3 * 10**9
    while len(recs) < N_VAL:
        r = make_random_instance(seed, cfg); seed += 1
        if r is not None: recs.append(r)
    save(f"val_simplex_{cost}", cost, [np.asarray(r["a"]) for r in recs], [np.asarray(r["b"]) for r in recs],
         ["simplex"] * N_VAL, ["simplex"] * N_VAL, [r["seed"] for r in recs],
         f"{N_VAL} validation instances, simplex marginals, seeds from {3 * 10**9}", folder="sets_val")
json.dump(manifest, open(f"{OUT}/sets/manifest.json", "w"), indent=2)
