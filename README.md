# ToR-OT: Learning Optimal Transport Plans Via Autoregressive Token Regression

Code for the NeurIPS 2026 paper *"Learning Optimal Transport Plans Via Autoregressive
Token Regression"* (Ivan Jaen Marquez, Takis Chytas, Vikas Singh — University of
Wisconsin–Madison).

ToR-OT casts discrete optimal transport as sequence generation: a transformer reads the
two marginals and the cost type and decodes the sparse optimal plan row by row as tokens.

## Repository layout

| Directory | Contents |
|---|---|
| [`torot/`](torot/) | **v2 (camera-ready)**: encoder–decoder model, exponent-first numeric tokenizer, significance-weighted loss, Hugging Face corpus generator, training and evaluation. See [`torot/README.md`](torot/README.md). |
| [`baselines/`](baselines/) | Every baseline in one place: POT and Sinkhorn, Meta OT and UNOT (official code, vendored, with adapters to this setting), ToR-OT v1. Each with its pinned environment. See [`baselines/README.md`](baselines/README.md). |
| [`reproduce/`](reproduce/) | Scripts that produce every camera-ready number, table and figure (evaluation sets, predictions, scoring, figures, GPU queue). See [`reproduce/README.md`](reproduce/README.md). |
| `experiments/` | Outputs (checkpoints, predictions, scores, figures); not tracked. |
| [`TF4OT/`](TF4OT/) | v1 code used for the submitted experiments (single-prefix decoder, row-wise datasets) plus the helper scripts, notebooks, assets and cluster files that produced the paper's figures. See [`TF4OT/README.md`](TF4OT/README.md). |
| `paper/` | Manuscript. |

## Cost types

Every instance has one of six cost types. The paper writes them as below; the code, the dataset (`cost_type`) and
the model's cost token use the code names. $L_2^p$ is the Euclidean distance to the power p, so in 1D
$L_2 = |x - y|$ (which on a line also equals $L_1$ and $L_\infty$). Each cost matrix is divided by its largest entry.

| Paper | Code / dataset name | $c(x, y)$ |
|---|---|---|
| $L_2$ | `euclidean` | $\lVert x - y\rVert_2$ |
| $L_2^{1/2}$ | `sqrt_euclidean` | $\lVert x - y\rVert_2^{1/2}$ |
| $L_2^2$ | `sqeuclidean` | $\lVert x - y\rVert_2^2$ |
| $L_2^3$ | `cubic_euclidean` | $\lVert x - y\rVert_2^3$ |
| Canberra | `canberra` | $\sum_d \lvert x_d - y_d\rvert / (\lvert x_d\rvert + \lvert y_d\rvert)$ |
| Gaussian | `gaussian` | $1 - \exp(-\lVert x - y\rVert_2^2 / 2\sigma^2)$, $\sigma = 0.2$ |

Definitions: [`torot/data/cost.py`](torot/data/cost.py). The submitted version listed $L_1$, $L_2$, $L_2^2$, $L_3$,
$L_\infty$ and Canberra; in 1D the first, second, fourth and fifth are all $|x - y|$, so the camera-ready corpus uses
six costs that differ in 1D.

## Quick start (v2)

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda create -y -n env_torot python=3.11
conda activate env_torot
pip install -e .
pytest torot/tests
```

Then follow [`torot/README.md`](torot/README.md) for data generation, training and evaluation.

## Citation

```bibtex
@inproceedings{jaenmarquez2026torot,
  title     = {Learning Optimal Transport Plans Via Autoregressive Token Regression},
  author    = {Jaen Marquez, Ivan and Chytas, Takis and Singh, Vikas},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). Vendored third-party code keeps its own license: `baselines/metaot/third_party/`
(Meta OT, CC BY-NC 4.0, non-commercial) and `baselines/unot/third_party/` (UNOT, MIT).
