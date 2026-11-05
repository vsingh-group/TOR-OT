"""Build the 2D evaluation sets (env_torot, repository root).

    python reproduce/build_sets_2d.py experiments/torot/camera_ready_2d [costs...] [--side 6]

Same file format as reproduce/build_sets.py (1D), plus the bin coordinates (``coords`` [n, 2], read by
``torot.data.supports.set_coords``), ``dim`` = 2 and ``side``. Grid: side x side bin centres on [0, 1]^2, x-major
(``torot.data.supports.grid_2d``); side 6 (36 bins) is the 2D setting of the paper. Costs: all six by default.

* ood2d_paper_<cost>  : the 2D OOD set: the seven named 2D distributions (SwissRoll, Mix8Gaussians,
                        StandardNormal, Moons, Circles, SCurve, UniformSquare), all 49 ordered pairs
                        (``torot.data.families.named_marginal_2d``, fixed seeds;
                        identical to the test_ood split of the 2D corpus at the same side).
* id_simplex2d_<cost> : 1,000 random instances, marginals uniform on the simplex (the 2D training marginals);
                        seeds from 4e9.
* sets_val/val_simplex2d_<cost> : 500 validation instances (model selection of the learned baselines); seeds from 5e9.

Seeds are disjoint from the 1D sets (1e9-3e9 + 1e6) and from the corpora (2D corpus: seed 2, i.e. 2e10 + offsets).
"""
import json, os, sys
import numpy as np, ot
from torot.data.cost import COST_TYPES, cost_matrix
from torot.data.families import NAMED_2D_FAMILIES, named_marginal_2d
from torot.data.generate import GenConfig, make_random_instance
from torot.data.supports import grid_2d

OUT, N_ID, N_VAL = sys.argv[1], 1000, 500
SIDE = int(sys.argv[sys.argv.index("--side") + 1]) if "--side" in sys.argv else 6
COSTS = tuple(a for i, a in enumerate(sys.argv[2:], 2) if not a.startswith("--") and sys.argv[i - 1] != "--side") or COST_TYPES
coords = grid_2d(SIDE)
N = SIDE * SIDE
os.makedirs(f"{OUT}/sets", exist_ok=True)
os.makedirs(f"{OUT}/sets_val", exist_ok=True)
manifest = json.load(open(f"{OUT}/sets/manifest.json")) if os.path.exists(f"{OUT}/sets/manifest.json") else {}


def solve(a, b, cost):
    C = cost_matrix(coords, coords, cost)
    P = ot.emd(a, b, C, numItermax=1_000_000)
    P[P < 1e-12] = 0.0
    return P, float((P * C).sum())


def save(name, cost, a, b, src, tgt, seeds, description, folder="sets"):
    plans, costs = zip(*[solve(x, y, cost) for x, y in zip(a, b)])
    np.savez(f"{OUT}/{folder}/{name}.npz", a=np.stack(a), b=np.stack(b), plan=np.stack(plans), ot_cost=np.array(costs),
             src=np.array(src), tgt=np.array(tgt), seed=np.array(seeds), cost=cost, n=N, dim=2, side=SIDE, coords=coords)
    if folder == "sets":
        manifest[name] = {"cost": cost, "instances": len(a), "side": SIDE, "description": description}
    print(f"{name:<30} {len(a):>5} instances", flush=True)


def random_set(cost, base, count):
    cfg = GenConfig(dim=2, side_min=SIDE, side_max=SIDE, cost_types=[cost], marginals="simplex")
    recs, seed = [], base
    while len(recs) < count:
        r = make_random_instance(seed, cfg); seed += 1
        if r is not None:
            recs.append(r)
    return recs


names = list(NAMED_2D_FAMILIES)
hists = {f: named_marginal_2d(f, SIDE) for f in names}
for cost in COSTS:
    pairs = [(fa, fb) for fa in names for fb in names]
    save(f"ood2d_paper_{cost}", cost, [hists[fa] for fa, _ in pairs], [hists[fb] for _, fb in pairs],
         [fa for fa, _ in pairs], [fb for _, fb in pairs], [-1] * len(pairs),
         f"2D OOD set: 7 named distributions x 49 ordered pairs, {SIDE} x {SIDE} grid")
    recs = random_set(cost, 4 * 10**9, N_ID)
    save(f"id_simplex2d_{cost}", cost, [np.asarray(r["a"]) for r in recs], [np.asarray(r["b"]) for r in recs],
         ["simplex"] * N_ID, ["simplex"] * N_ID, [r["seed"] for r in recs],
         f"{N_ID} random instances, simplex marginals, {SIDE} x {SIDE} grid, seeds from {4 * 10**9}")
    recs = random_set(cost, 5 * 10**9, N_VAL)
    save(f"val_simplex2d_{cost}", cost, [np.asarray(r["a"]) for r in recs], [np.asarray(r["b"]) for r in recs],
         ["simplex"] * N_VAL, ["simplex"] * N_VAL, [r["seed"] for r in recs],
         f"{N_VAL} validation instances, simplex marginals, seeds from {5 * 10**9}", folder="sets_val")
json.dump(manifest, open(f"{OUT}/sets/manifest.json", "w"), indent=2)
