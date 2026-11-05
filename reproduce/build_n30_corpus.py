"""Corpus of the ablation testbed: torot_data_1M restricted to n <= 30, optionally with a fraction of its train split.

    python reproduce/build_n30_corpus.py experiments/torot/ablations/corpus_n30                       (from the Hub)
    python reproduce/build_n30_corpus.py experiments/torot/ablations/corpus_n30_25pct \\
        --from-local experiments/torot/ablations/corpus_n30 --train-fraction 0.25

The 1M corpus' instances are i.i.d. (one seed each), so the first fraction of the train split is a uniform subset;
val / test splits are kept whole so that every ablation is evaluated on the same instances. (env_torot, CPU)
"""
import argparse
import json
import os

from datasets import DatasetDict, load_dataset, load_from_disk

REPO, REVISION = "ivanjaenm/torot_data_1M", "01a6710ce743d34b1a0bfed10b8b7d2e9bdff3e2"

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--from-local", default=None, help="an existing n <= 30 corpus (save_to_disk) instead of the Hub")
ap.add_argument("--train-fraction", type=float, default=1.0)
ap.add_argument("--n-max", type=int, default=30)
args = ap.parse_args()

if args.from_local:
    ds, source = load_from_disk(args.from_local), json.load(open(f"{args.from_local}/SOURCE.json")) if os.path.exists(f"{args.from_local}/SOURCE.json") else {"from": args.from_local}
else:
    ds = load_dataset(REPO, revision=REVISION)
    ds = DatasetDict({k: v.filter(lambda n: n <= args.n_max, input_columns="n") for k, v in ds.items()})
    source = {"repo": REPO, "revision": REVISION, "filter": f"n <= {args.n_max}"}
if args.train_fraction < 1.0:
    ds["train"] = ds["train"].select(range(int(round(args.train_fraction * len(ds["train"])))))
    source = {**source, "train_fraction": args.train_fraction}
ds.save_to_disk(args.out)
json.dump({**source, "sizes": {k: len(v) for k, v in ds.items()}}, open(f"{args.out}/SOURCE.json", "w"), indent=2)
print(args.out, {k: len(v) for k, v in ds.items()})
