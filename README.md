# ToR-OT: Learning Optimal Transport Plans Via Autoregressive Token Regression

🤗 **Models:** [`torot-1d`](https://huggingface.co/ivanjaenm/torot-1d) · [`torot-2d`](https://huggingface.co/ivanjaenm/torot-2d) &nbsp;|&nbsp; 🤗 **Datasets:** [`torot_data_5M`](https://huggingface.co/datasets/ivanjaenm/torot_data_5M) · [`torot_data_4M_2d`](https://huggingface.co/datasets/ivanjaenm/torot_data_4M_2d) · [`torot_data_1M`](https://huggingface.co/datasets/ivanjaenm/torot_data_1M)

ToR-OT (**To**ken **R**egression for **O**ptimal **T**ransport) casts discrete optimal transport as sequence generation: a transformer reads the two marginals and the cost and decodes the sparse optimal plan row by row as tokens.

## Installation

```bash
conda create -y -n env_torot python=3.11
conda activate env_torot
pip install -e .                  # installs torot/requirements.txt and makes `torot` importable
pytest torot/tests                # CPU, a few minutes
```

`torot/requirements.txt` is the only dependency list (torch, POT, datasets, scipy,
omegaconf). A CPU-only machine used just for data generation needs only `pip install numpy scipy POT datasets huggingface_hub`. The baselines use their own environments
([`baselines/README.md`](baselines/README.md)).

## Quick start

```python
import numpy as np, torch, ot
from torot.evaluate import load_model
from torot.decode import predict_plans
from torot.data.cost import COST_TYPES, cost_matrix
from torot.data.supports import grid_1d

device = torch.device("cuda")
model, tok, cfg = load_model("ivanjaenm/torot-1d", device)              # downloads the weights

n = 30
x = grid_1d(n)                                           # bin centres on [0, 1], shape [n, 1]
rng = np.random.default_rng(0)
a, b = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(n))
inst = {"n": n, "m": n, "dim": 1, "cost_type": COST_TYPES.index("sqeuclidean"),
        "coords_src": x, "coords_tgt": x, "a": a, "b": b}
plan = predict_plans(model, tok, [inst], device, last_row="model")[0]   # dense n x m plan

C = cost_matrix(x, x, "sqeuclidean")
print((plan * C).sum() / (ot.emd(a, b, C) * C).sum() - 1)               # relative cost gap to the exact plan
```

## How ToR-OT model works

* **Input.** An encoder reads one token per bin (coordinates and mass) of both marginals, plus a cost-type token, so
  any support sizes `n`, `m` are accepted.
* **Output.** The plan is generated one row at a time, and each row is conditioned on the rows before it.
* **Tokens.** A row is `BOS (⟨tgt j⟩ E-k d1 d2 d3 d4)* EOS`: per nonzero entry a pointer to the target bin, then the
  mass in scientific notation (exponent, four mantissa digits).
* **Decoding** (`torot/decode.py`): greedy, with a grammar mask that allows only tokens that can legally follow, a KV cache, and a stop at `EOS`.
* **Training.** One example per (instance, row), built on the fly from the corpus. The loss is a cross-entropy
  weighted by significance (exponent, then d1 > d2 > …) with ordinal soft labels; transposed instances are used as
  augmentation.

## Cost types

Every instance has one cost type. The paper writes them as below; the code, the dataset (`cost_type`) and
the model's cost token use the code names.

| Paper | Code / dataset name | $c(x, y)$ |
|---|---|---|
| $L_2$ | `euclidean` | $\lVert x - y\rVert_2$ |
| $L_2^{1/2}$ | `sqrt_euclidean` | $\lVert x - y\rVert_2^{1/2}$ |
| $L_2^2$ | `sqeuclidean` | $\lVert x - y\rVert_2^2$ |
| $L_2^3$ | `cubic_euclidean` | $\lVert x - y\rVert_2^3$ |
| Canberra | `canberra` | $\sum_d \lvert x_d - y_d\rvert / (\lvert x_d\rvert + \lvert y_d\rvert)$ |
| Gaussian | `gaussian` | $1 - \exp(-\lVert x - y\rVert_2^2 / 2\sigma^2)$, $\sigma = 0.2$ |

Definitions: [`torot/data/cost.py`](torot/data/cost.py). The six costs are pairwise distinct in 1D; `euclidean` has
non-unique optimal plans there.

## Data

The synthetic corpora are generated on CPU (the only step that runs POT) and pushed to the Hub. The commands of the three
corpora:

```bash
python -m torot.data.generate --out_dir ./torot_data_5M --repo_id <user>/torot_data_5M --push \
    --train 5000000 --val 2500 --test_id 5000 --test_extrap 5000 --seed 0 --workers 32
python -m torot.data.generate --dim 2 --out_dir ./torot_data_4M_2d --repo_id <user>/torot_data_4M_2d --push \
    --train 4000000 --val 2500 --test_id 5000 --test_extrap 5000 --seed 3 --workers 8 --empty_bins 0.5
python -m torot.data.generate --out_dir ./torot_data_1M --repo_id <user>/torot_data_1M --push \
    --train 1000000 --val 2500 --test_id 5000 --test_extrap 5000 --seed 0 --workers 12
```

* **Instances.** A regular grid of bin centres on [0, 1] (1D, n ∈ [5, 100]) or a k x k grid on [0, 1]² (2D,
  k ∈ [3, 10]); n = m; one of the six costs; the exact plan from POT's network simplex (`ot.emd`). Marginals are
  drawn uniformly at random. One record holds a whole plan.
* **Splits.** `train`, `val`, `test_id` (same distribution, disjoint seeds), `test_extrap` (larger supports: 1D
  n ∈ [120, 200], 2D k ∈ [11, 14]) and `test_ood`, the named distributions of the paper's OOD evaluation (1D: the 144
  ordered pairs of 12 densities at n = 30 and 100; 2D: the 49 pairs of 7 distributions at k = 6 and 10), for every
  cost.
* Every instance is generated from its own seed, so any record can be regenerated independently of the number of
  workers.

## Training

The configs of the two models of the paper (157.6M parameters each, row decoding):

| config | model | corpus | trained on |
|---|---|---|---|
| `torot/configs/torot_1d.yaml` | `ivanjaenm/torot-1d` | `ivanjaenm/torot_data_5M` | 8 GPUs, 203,000 steps |
| `torot/configs/torot_2d.yaml` | `ivanjaenm/torot-2d` | `ivanjaenm/torot_data_4M_2d` | 2 GPUs, 627,000 steps |

```bash
torchrun --nproc_per_node 8 -m torot.train --config torot/configs/torot_1d.yaml   # writes experiments/torot/row_M_5M
python -m torot.train --config torot/configs/tiny.yaml                            # a quick sanity run
```

## Evaluation

```bash
python -m torot.evaluate --checkpoint ivanjaenm/torot-1d --split test_id --last_row model --out test_id.json
```

Reports the signed cost gap and the marginal violations and plan-level metrics (MAPE, weighted MAPE, support precision / recall), mean and median.

## Reproducing the paper

[`reproduce/README.md`](reproduce/README.md) gives the evaluation sets to each figure and table of the
paper; [`baselines/README.md`](baselines/README.md) describes the baselines
(POT, Sinkhorn, Meta OT, UNOT, NeuralOT) and how they were trained and selected.

## Models and data

| | Hugging Face | Used for |
|---|---|---|
| 1D model (157.6M parameters) | [`ivanjaenm/torot-1d`](https://huggingface.co/ivanjaenm/torot-1d) | all 1D results |
| 2D model (157.6M parameters) | [`ivanjaenm/torot-2d`](https://huggingface.co/ivanjaenm/torot-2d) | Fig. 4C–D, Fig. 5, App. F |
| 1D training corpus | [`ivanjaenm/torot_data_5M`](https://huggingface.co/datasets/ivanjaenm/torot_data_5M) | training the 1D model |
| 2D training corpus | [`ivanjaenm/torot_data_4M_2d`](https://huggingface.co/datasets/ivanjaenm/torot_data_4M_2d) | training the 2D model |
| Ablation corpus | [`ivanjaenm/torot_data_1M`](https://huggingface.co/datasets/ivanjaenm/torot_data_1M) | App. G (small testbed) |

## Repository layout

| Directory | Contents |
|---|---|
| [`torot/`](torot/) | The model: encoder–decoder transformer, numeric tokenizer, loss, corpus generator, training, decoding and evaluation; configs in `torot/configs/`, unit tests in `torot/tests/`. |
| [`baselines/`](baselines/) | Every baseline, one folder each: `pot/` (exact OT and Sinkhorn), `metaot/`, `unot/` and `neuralot/` (official code, vendored, with adapters to this setting). |
| [`reproduce/`](reproduce/) | Scripts that produce every number, table and figure of the paper, and the `pip freeze` of each environment used. |
| `experiments/` | Outputs (checkpoints, predictions, scores, figures); created by the scripts, not tracked. |

## Citation

```bibtex
@inproceedings{jaenmarquez2026torot,
  title     = {Learning Optimal Transport Plans Via Autoregressive Token Regression},
  author    = {Jaen Marquez, Ivan and Chytas, Sotirios Panagiotis and Singh, Vikas},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). Vendored third-party code keeps its own license: `baselines/metaot/third_party/`
(Meta OT, CC BY-NC 4.0, non-commercial), `baselines/unot/third_party/` (UNOT, MIT) and `baselines/neuralot/third_party/`
(Neural Optimal Transport, MIT).
