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
| [`TF4OT/`](TF4OT/) | v1 code used for the submitted experiments (single-prefix decoder, row-wise datasets) plus the helper scripts, notebooks, assets and cluster files that produced the paper's figures. See [`TF4OT/README.md`](TF4OT/README.md). |
| `paper/` | Manuscript. |

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

MIT — see [LICENSE](LICENSE).
