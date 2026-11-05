"""Export a trained ToR-OT model (and optionally the learned baselines) for the Hugging Face Hub (env_torot, CPU).

    # 1D model (all 1D results)
    python reproduce/export_to_hub.py --kind 1d --checkpoint experiments/torot/row_M_5M/best.pt \\
        --out experiments/hub/torot-1d --repo-id ivanjaenm/torot-1d --baselines experiments/baselines \\
        [--data-curve-checkpoints]
    # 2D model (Fig. 4C-D, Fig. 5, App. F)
    python reproduce/export_to_hub.py --kind 2d --checkpoint <2D run>/best.pt \\
        --out experiments/hub/torot-2d --repo-id ivanjaenm/torot-2d --baselines experiments/baselines_2d \\
        --dataset ivanjaenm/torot_data_4M_2d@<revision>
    hf upload ivanjaenm/torot-1d experiments/hub/torot-1d .                                # by the authors

Writes model.safetensors (EMA weights, float32, as used for every reported number), config.yaml (with data.repo_id set
to --dataset), README.md (model card), SHA256SUMS. Options: --data-curve-checkpoints adds checkpoints/step<k>/ (EMA
weights of the training checkpoints used for the data curve); --baselines adds baselines/<method>/<run>/ (the Meta OT /
UNOT runs selected on validation, one per cost). Load with torot.evaluate.load_model("<folder or repo id>", device).
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

DATASETS = {"1d": "ivanjaenm/torot_data_5M@54892f9763ce5b9b7237a2790aed8fff85e98baf",
            "2d": "ivanjaenm/torot_data_4M_2d@1ff4a4a6329fa5c822f77d92b1c2ae850afedd9b"}
RESULTS = {"1d": "experiments/torot/camera_ready/eval_row_M_5M", "2d": None}

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=("1d", "2d"), default="1d")
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--repo-id", required=True)
ap.add_argument("--dataset", default=None, help="training corpus as repo@revision (default: the corpus of --kind)")
ap.add_argument("--results", default=None,
                help="folder with torot.evaluate outputs eval_<split>_lastrow.json for the card (default for 1d: "
                     "experiments/torot/camera_ready/eval_row_M_5M; none for 2d)")
ap.add_argument("--code-url", default="https://github.com/vsingh-group/TOR-OT")
ap.add_argument("--data-curve-checkpoints", action="store_true")
ap.add_argument("--baselines", default=None)
args = ap.parse_args()
dataset = args.dataset or DATASETS[args.kind]
results = args.results or RESULTS[args.kind]
repo, _, rev = dataset.partition("@")
os.makedirs(args.out, exist_ok=True)
run = os.path.dirname(args.checkpoint)


def export(ckpt_path, folder):
    os.makedirs(folder, exist_ok=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    save_file({k: v.contiguous() for k, v in ck["ema"].items()}, f"{folder}/model.safetensors",
              metadata={"step": str(ck.get("step")), "weights": "ema", "source": os.path.basename(ckpt_path)})
    cfg = OmegaConf.load(f"{run}/config.yaml")
    cfg.data.repo_id = repo                       # the corpus' released name (it may have been renamed after training)
    OmegaConf.save(cfg, f"{folder}/config.yaml")
    return ck.get("step"), sum(v.numel() for v in ck["ema"].values())


step, n_params = export(args.checkpoint, args.out)
if args.data_curve_checkpoints:
    for path in sorted(glob.glob(f"{run}/ckpt_*.pt")):
        k = int(os.path.basename(path)[5:-3])
        if k % 25000 == 0:
            export(path, f"{args.out}/checkpoints/step{k:06d}")
if args.baselines:
    sys.path.insert(0, os.getcwd())
    from baselines.common import select_run
    from torot.data.cost import COST_TYPES
    for method, files in (("metaot", ("config.json", "metrics.json", "params.msgpack")), ("unot", ("config.json", "metrics.json", "predictor.pt", "generator.pt"))):
        runs = sorted(glob.glob(f"{args.baselines}/{method}/*"))
        for cost in COST_TYPES:
            best, _ = select_run(runs, cost)
            dest = f"{args.out}/baselines/{method}/{os.path.basename(best)}"
            os.makedirs(dest, exist_ok=True)
            for f in files:
                shutil.copy(f"{best}/{f}", dest)

metrics = {}
for split in ("test_id", "test_ood", "test_extrap"):
    path = f"{results}/eval_{split}_lastrow.json" if results else ""
    if path and os.path.exists(path):
        s = json.load(open(path))["summary"]
        metrics[split] = (s["instances"], s["wmape/median"], s["wmape/mean"], s["support_recall/mean"], s["max_violation/median"])
table = ""
if metrics:
    rows = "\n".join(f"| {k} | {v[0]} | {v[1]:.4f} | {v[2]:.4f} | {v[3]:.3f} | {v[4]:.1e} |" for k, v in metrics.items())
    table = ("## Results on the corpus' test splits (weighted MAPE = the paper's MAPE)\n\n"
             "| split | instances | wMAPE (median) | wMAPE (mean) | support recall | max violation (median) |\n"
             "|---|---|---|---|---|---|\n" + rows + "\n")

if args.kind == "1d":
    title = f"ToR-OT 1D model ({n_params / 1e6:.1f}M parameters)"
    used_for = "every 1D result of the paper"
    data_desc = ("1D histograms uniform on the probability simplex on a regular grid of n = 5-100 bins in [0, 1], six "
                 "cost types (euclidean, sqrt_euclidean, sqeuclidean, cubic_euclidean, canberra, gaussian)")
    example = '''n = 30
x = grid_1d(n)                                           # bin centres on [0, 1], shape [n, 1]
rng = np.random.default_rng(0)
a, b = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(n))
inst = {"n": n, "m": n, "dim": 1, "cost_type": COST_TYPES.index("sqeuclidean"),
        "coords_src": x, "coords_tgt": x, "a": a, "b": b}'''
    imports = "from torot.data.supports import grid_1d"
else:
    title = f"ToR-OT 2D model ({n_params / 1e6:.1f}M parameters)"
    used_for = "the 2D results of the paper (Fig. 4C-D, Fig. 5, App. F)"
    data_desc = ("2D histograms on k x k grids in [0, 1]^2 (k = 3-10, n = 9-100 bins), six cost types; half of the "
                 "marginals have empty bins (10-70 % of the bins set to zero)")
    example = '''k = 6
x = grid_2d(k)                                           # k*k bin centres on [0, 1]^2, shape [k*k, 2]
n = k * k
rng = np.random.default_rng(0)
a, b = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(n))
inst = {"n": n, "m": n, "dim": 2, "cost_type": COST_TYPES.index("sqeuclidean"),
        "coords_src": x, "coords_tgt": x, "a": a, "b": b}'''
    imports = "from torot.data.supports import grid_2d"

baselines_note = ("`baselines/`: the Meta OT and UNOT runs used in the paper (one per cost, selected on validation; code in "
                  "`baselines/` of the repository).\n") if args.baselines else ""
curve_note = ("`checkpoints/step*/`: EMA weights of the intermediate checkpoints (training-data curve).\n"
              if args.data_curve_checkpoints else "")
open(f"{args.out}/README.md", "w").write(f"""---
license: mit
library_name: torot
tags: [optimal-transport, transformer, token-regression]
datasets: [{repo}]
---

# {title}

Model of *Learning Optimal Transport Plans Via Autoregressive Token Regression* (NeurIPS 2026), used for {used_for}.
An encoder reads the two histograms and the cost type; a decoder writes the optimal transport plan row by row as
tokens (target-bin pointer, exponent, four mantissa digits). Code: {args.code_url}

* Weights: exponential moving average after {step} training steps (`model.safetensors`, float32); architecture in
  `config.yaml`.
* Training data: [`{repo}`](https://huggingface.co/datasets/{repo}) at revision `{rev}`: {data_desc}.
* Decoding used for the paper: greedy, grammar-constrained, row by row, last row decoded by the model
  (`last_row="model"`).

```python
import numpy as np, torch, ot
from torot.evaluate import load_model
from torot.decode import predict_plans
from torot.data.cost import COST_TYPES, cost_matrix
{imports}

device = torch.device("cuda")
model, tok, cfg = load_model("{args.repo_id}", device)

{example}
plan = predict_plans(model, tok, [inst], device, last_row="model")[0]           # dense n x m plan

C = cost_matrix(x, x, "sqeuclidean")
print((plan * C).sum() / (ot.emd(a, b, C) * C).sum() - 1)                       # relative cost gap to the exact plan
```

{table}
{baselines_note}{curve_note}
## Citation

```bibtex
@inproceedings{{jaenmarquez2026torot,
  title     = {{Learning Optimal Transport Plans Via Autoregressive Token Regression}},
  author    = {{Jaen Marquez, Ivan and Chytas, Sotirios Panagiotis and Singh, Vikas}},
  booktitle = {{Advances in Neural Information Processing Systems (NeurIPS)}},
  year      = {{2026}}
}}
```
""")
with open(f"{args.out}/SHA256SUMS", "w") as fp:
    for path in sorted(glob.glob(f"{args.out}/**/*", recursive=True)):
        if os.path.isfile(path) and not path.endswith("SHA256SUMS"):
            fp.write(f"{hashlib.sha256(open(path, 'rb').read()).hexdigest()}  {os.path.relpath(path, args.out)}\n")
print(f"exported {n_params / 1e6:.1f}M parameters (step {step}) to {args.out}")
