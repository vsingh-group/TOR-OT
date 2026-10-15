"""Generate the ToR-OT v2 corpus and (optionally) push it to the Hugging Face Hub.

Example (CPU box):

    python -m torot.data.generate \\
        --out_dir ./otdata_v2 --repo_id ivanjaenm/otdata_v2 --push \\
        --train 2000000 --val 5000 --test_id 10000 --test_extrap 5000 \\
        --seed 0 --workers 16

Instances: 1D regular grid, marginals drawn uniformly from the probability simplex
(``--marginals simplex``, the default) or with v1's normalised uniform weights
(``--marginals normalized_uniform``, reproduces ``torot_data_1M_v2``), exact plan from POT's
network simplex; the support size ``n`` varies per instance.
Every instance is generated from its own seed, so any record can be regenerated
independently of the number of workers or the chunking.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np
import ot
from datasets import Dataset, DatasetDict

from torot.data.cost import COST_TYPES, cost_matrix
from torot.data.families import MARGINAL_SAMPLERS, NAMED_1D_FAMILIES, named_marginal
from torot.data.schema import DATASET_CARD_FRONT_MATTER, DATASET_CARD_TEMPLATE, FEATURES
from torot.data.supports import grid_1d

PLAN_EPS = 1e-12          # plan entries below this are numerical zeros
FEASIBILITY_TOL = 1e-7    # discard instances whose plan violates the marginals by more


@dataclass
class GenConfig:
    n_min: int = 5                        # support size range for train / val / test_id
    n_max: int = 100
    extrap_n_min: int = 120               # support size range for test_extrap
    extrap_n_max: int = 200
    equal_sizes: bool = True              # n == m as in v1; False samples n and m independently
    marginals: str = "simplex"            # "simplex" (Dirichlet(1)) or "normalized_uniform" (v1 / torot_data_1M_v2)
    cost_types: List[str] = field(default_factory=lambda: list(COST_TYPES))
    ood_sizes: List[int] = field(default_factory=lambda: [30, 100])

    def __post_init__(self) -> None:
        if self.marginals not in MARGINAL_SAMPLERS:
            raise ValueError(f"marginals must be one of {list(MARGINAL_SAMPLERS)}, got {self.marginals!r}")


# --------------------------------------------------------------------------- single instance
def solve_instance(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> Optional[tuple[np.ndarray, float]]:
    """Exact LP solution via POT's network simplex; None if the plan fails the sanity checks."""
    plan = ot.emd(a, b, C, numItermax=1_000_000)
    plan = np.where(plan > PLAN_EPS, plan, 0.0)
    if plan.min() < 0.0:
        return None
    if np.abs(plan.sum(1) - a).max() > FEASIBILITY_TOL or np.abs(plan.sum(0) - b).max() > FEASIBILITY_TOL:
        return None
    return plan, float((plan * C).sum())


def _record(
    seed: int,
    coords_src: np.ndarray,
    coords_tgt: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    cost_type: str,
    family_src: str,
    family_tgt: str,
) -> Optional[dict]:
    C = cost_matrix(coords_src, coords_tgt, cost_type)
    solved = solve_instance(a, b, C)
    if solved is None:
        return None
    plan, cost = solved
    rows, cols = np.nonzero(plan)
    return {
        "n": int(a.shape[0]),
        "m": int(b.shape[0]),
        "dim": int(coords_src.shape[1]),
        "cost_type": COST_TYPES.index(cost_type),
        "family_src": family_src,
        "family_tgt": family_tgt,
        "coords_src": coords_src.astype(np.float64).tolist(),
        "coords_tgt": coords_tgt.astype(np.float64).tolist(),
        "a": a.tolist(),
        "b": b.tolist(),
        "plan_rows": rows.astype(np.int32).tolist(),
        "plan_cols": cols.astype(np.int32).tolist(),
        "plan_vals": plan[rows, cols].tolist(),
        "ot_cost": cost,
        "seed": int(seed),
    }


def make_random_instance(seed: int, cfg: GenConfig, extrapolate: bool = False) -> Optional[dict]:
    """One random instance, fully determined by ``seed`` and ``cfg``."""
    rng = np.random.default_rng(seed)
    lo, hi = (cfg.extrap_n_min, cfg.extrap_n_max) if extrapolate else (cfg.n_min, cfg.n_max)
    n = int(rng.integers(lo, hi + 1))
    m = n if cfg.equal_sizes else int(rng.integers(lo, hi + 1))
    xs, xt = grid_1d(n), grid_1d(m)
    sample = MARGINAL_SAMPLERS[cfg.marginals]
    a, b = sample(rng, n), sample(rng, m)
    cost_type = cfg.cost_types[int(rng.integers(len(cfg.cost_types)))]
    return _record(seed, xs, xt, a, b, cost_type, cfg.marginals, cfg.marginals)


def make_ood_instances(cfg: GenConfig) -> Iterator[dict]:
    """Evaluation only: all ordered pairs of the named 1D densities, per OOD size and cost type."""
    names = list(NAMED_1D_FAMILIES)
    seed = 0
    for n, cost_type, (fa, fb) in itertools.product(cfg.ood_sizes, cfg.cost_types, itertools.product(names, names)):
        xs = grid_1d(n)
        rec = _record(seed, xs, xs, named_marginal(fa, xs), named_marginal(fb, xs), cost_type, fa, fb)
        seed += 1
        if rec is not None:
            yield rec


# --------------------------------------------------------------------------- sharded generation
def _random_records(shard_seeds: List[int], counts: List[int], cfg_dict: dict, extrapolate: bool) -> Iterator[dict]:
    """Generator consumed by ``Dataset.from_generator`` (sharded over ``shard_seeds``)."""
    cfg = GenConfig(**cfg_dict)
    for shard_seed, count in zip(shard_seeds, counts):
        produced = 0
        seed = shard_seed
        while produced < count:
            rec = make_random_instance(seed, cfg, extrapolate)
            seed += 1
            if rec is not None:
                produced += 1
                yield rec


def _ood_records(cfg_dict: dict) -> Iterator[dict]:
    """Generator consumed by ``Dataset.from_generator`` for the OOD split."""
    yield from make_ood_instances(GenConfig(**cfg_dict))


def build_random_split(
    num_instances: int,
    base_seed: int,
    cfg: GenConfig,
    workers: int,
    extrapolate: bool = False,
    chunk: int = 2000,
) -> Dataset:
    """Generate ``num_instances`` records in parallel shards; seeds never overlap across shards."""
    num_shards = max(1, -(-num_instances // chunk))
    counts = [chunk] * num_shards
    counts[-1] = num_instances - chunk * (num_shards - 1)
    # each shard may need a few extra seeds for discarded instances; reserve 2x its count
    shard_seeds = [base_seed + 2 * chunk * s for s in range(num_shards)]
    return Dataset.from_generator(
        _random_records,
        features=FEATURES,
        gen_kwargs={
            "shard_seeds": shard_seeds,
            "counts": counts,
            "cfg_dict": dataclasses.asdict(cfg),
            "extrapolate": extrapolate,
        },
        num_proc=min(workers, num_shards),
    )


def build_ood_split(cfg: GenConfig) -> Dataset:
    return Dataset.from_generator(
        _ood_records,
        features=FEATURES,
        gen_kwargs={"cfg_dict": dataclasses.asdict(cfg)},
    )


def upload_dataset_card(repo_id: str, body: str) -> None:
    """Upload the card, keeping the YAML front matter (``dataset_info``) written by ``push_to_hub``."""
    from huggingface_hub import HfApi, hf_hub_download

    header = DATASET_CARD_FRONT_MATTER
    try:
        existing = open(hf_hub_download(repo_id, "README.md", repo_type="dataset")).read()
        end = existing.find("\n---", 3)
        if existing.startswith("---") and end != -1:
            header = existing[: end + 4] + "\n"
    except Exception:  # no README yet
        pass
    HfApi().upload_file(
        path_or_fileobj=(header + body).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )


# --------------------------------------------------------------------------- CLI
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True, help="local directory for save_to_disk")
    p.add_argument("--repo_id", default=None, help="Hugging Face dataset repo, e.g. user/otdata_v2")
    p.add_argument("--push", action="store_true", help="push to the Hub after generation")
    p.add_argument("--private", action="store_true")
    p.add_argument("--train", type=int, default=2_000_000)
    p.add_argument("--val", type=int, default=5_000)
    p.add_argument("--test_id", type=int, default=10_000)
    p.add_argument("--test_extrap", type=int, default=5_000)
    p.add_argument("--no_ood", action="store_true", help="skip the named-density OOD evaluation split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--marginals", default=None, choices=sorted(MARGINAL_SAMPLERS),
                   help="marginal sampler (default: GenConfig.marginals = simplex)")
    p.add_argument("--config_json", default=None, help="JSON file overriding GenConfig fields")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = GenConfig()
    if args.config_json:
        with open(args.config_json) as f:
            cfg = GenConfig(**{**dataclasses.asdict(cfg), **json.load(f)})
    if args.marginals is not None:
        cfg = dataclasses.replace(cfg, marginals=args.marginals)

    # disjoint seed ranges per split (each split reserves 2 seeds per requested instance)
    offsets = {"train": 0, "val": 2 * args.train, "test_id": 2 * (args.train + args.val)}
    offsets["test_extrap"] = 2 * (args.train + args.val + args.test_id)
    base = args.seed * 10**10

    splits = {
        "train": build_random_split(args.train, base + offsets["train"], cfg, args.workers),
        "val": build_random_split(args.val, base + offsets["val"], cfg, args.workers),
        "test_id": build_random_split(args.test_id, base + offsets["test_id"], cfg, args.workers),
        "test_extrap": build_random_split(args.test_extrap, base + offsets["test_extrap"], cfg, args.workers, extrapolate=True),
    }
    if not args.no_ood:
        splits["test_ood"] = build_ood_split(cfg)

    dsd = DatasetDict(splits)
    dsd.save_to_disk(args.out_dir)

    splits_table = "\n".join(["| split | instances |", "|---|---|"] + [f"| `{k}` | {len(v)} |" for k, v in dsd.items()])
    card = DATASET_CARD_TEMPLATE.format(
        cost_types=", ".join(f"`{c}`" for c in cfg.cost_types),
        splits_table=splits_table,
        config_json=json.dumps(dataclasses.asdict(cfg), indent=2),
        command=" ".join(["python -m torot.data.generate"] + sys.argv[1:]),
    )
    with open(f"{args.out_dir}/README.md", "w") as f:
        f.write(DATASET_CARD_FRONT_MATTER + card)

    if args.push:
        if not args.repo_id:
            raise SystemExit("--push requires --repo_id")
        dsd.push_to_hub(args.repo_id, private=args.private)
        upload_dataset_card(args.repo_id, card)
        print(f"pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
