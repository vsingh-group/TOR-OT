# ToR-OT v2

Encoder–decoder implementation of *Learning Optimal Transport Plans Via Autoregressive
Token Regression* (NeurIPS 2026). The legacy single-prefix model lives in `TF4OT/`.

## What is new relative to v1

| | v1 (`TF4OT/`) | v2 (`torot/`) |
|---|---|---|
| Conditioning | one MLP prefix token over `[a, b, i/n]`, fixed `n` | set encoder over `n + m` bin tokens (coordinates + masses) + cost-type token; any `n`, `m` |
| Rows | decoded independently | whole plan decoded row-major; every row conditions on the previous ones |
| Indices | digit strings | row index implicit (`ROW` count); column index as a pointer over target-bin encoder states (size-agnostic) |
| Masses | fixed-point `0.dddd` | exponent-first scientific notation `<E-k> d1 d2 d3 d4` (uniform relative precision) |
| Loss | token cross-entropy | significance-weighted (exponent, d1 > d2 > …) cross-entropy with ordinal soft labels |
| Symmetries | — | transpose augmentation and test-time transpose averaging; permutation equivariance by construction |
| Decoding | sampling, no cache, fixed length | greedy, grammar-constrained, KV cache, stops at EOS (see [Decoding](#decoding)) |
| Training | Adam, constant LR, fp32 | AdamW, warmup + cosine, grad clipping, bf16, EMA, best-val checkpoint |
| Data | one fixed `n` per dataset, rows stored independently | same protocol (1D grid, uniformly random marginals, POT plans) but one HF corpus with `n` ∈ [5, 100] and whole plans per record |

## Decoding

Generation in `torot/decode.py` differs from v1 in four independent ways:

* **Greedy.** The next token is the argmax of the logits. v1 sampled from the softmax
  (`torch.multinomial` at temperature 0.1), which adds noise to a numeric target and makes
  results non-deterministic.
* **Grammar-constrained.** The plan is serialised as
  `BOS (ROW (⟨tgt j⟩ E-k d1 d2 d3 d4)*)ⁿ EOS`: one `ROW` block per source bin in order (the
  row index is implicit, the number of `ROW` tokens so far), and per entry a pointer to the
  target bin, the exponent and four mantissa digits. Before the argmax, every token that
  cannot legally follow is masked out, so the output always parses; v1 needed string-repair
  heuristics and scored unparseable rows as all-zero. The mask also enforces the
  convention (exactly `n` row blocks, increasing target indices within a row, no leading
  `0` mantissa digit). It does *not* enforce marginal feasibility.
* **KV cache.** Each decoder layer stores the keys/values of the tokens generated so far,
  so a new token costs one layer pass over one token instead of a full pass over the whole
  prefix (v1). Cross-attention keys/values over the encoder are computed once. The result is
  identical to the uncached forward pass (checked by a unit test).
* **EOS.** Generation stops when every sequence in the batch has emitted `EOS`; v1 always
  generated `max_seq_len` tokens and truncated at `]` afterwards.

## Environment

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda create -y -n env_torot python=3.11
conda activate env_torot
pip install -e .                  # installs torot/requirements.txt and makes `torot` importable
```

`torot/requirements.txt` is the only dependency list for v2 (torch from the default PyPI
CUDA wheels, POT, datasets, scipy, omegaconf); the legacy v1 pins live in
`TF4OT/requirements.txt`. No `flash-attn` build is needed: attention
uses `torch.nn.functional.scaled_dot_product_attention`. On a CPU-only machine used just
for data generation, `pip install numpy scipy POT datasets huggingface_hub` is enough,
since `torot.data.generate` does not import torch.

Verify the installation with the unit tests (CPU, a few seconds):

```bash
pytest torot/tests
```

## Data

Generate the corpus on a CPU machine and push it to the Hub (this is the only step that
runs POT):

```bash
python -m torot.data.generate --out_dir ./otdata_v2 --repo_id ivanjaenm/otdata_v2 --push \
    --train 2000000 --val 5000 --test_id 10000 --test_extrap 5000 --seed 0 --workers 16
```

The instance protocol is the same as v1: 1D regular grid on [0, 1], marginals sampled
uniformly at random (no parametric family), all six cost types, exact plan from POT's
network simplex. The only new axis is the support size, `n` ∈ [5, 100] per instance
(`n = m`; `equal_sizes: false` in `GenConfig` samples them independently). One record
holds a whole plan, not a row.

Splits: `train`, `val`, `test_id` (same protocol, disjoint seeds), `test_extrap`
(`n` ∈ [120, 200], for extrapolation over the number of bins), and `test_ood` — the
evaluation-only split of 144 ordered pairs of the 12 named 1D densities used for the
paper's OOD evaluation, at 30 and 100 bins and all cost types (`--no_ood` skips it).
Every instance is reproducible from its `seed`.

## Training

```bash
python -m torot.train --config torot/configs/tiny.yaml                       # sanity run
torchrun --nproc_per_node 4 -m torot.train --config torot/configs/default.yaml
```

Overrides use dot syntax, e.g. `optim.lr=2e-4 run.out_dir=experiments/torot/lr2e-4`.
Checkpoints (`best.pt`, `last.pt`) contain raw and EMA weights; `config.yaml` is saved beside them.

## Evaluation

```bash
python -m torot.evaluate --checkpoint experiments/torot/base/best.pt --split test_id --symmetric --out test_id.json
```

Reports solution-invariant metrics (signed cost gap, marginal violations) and plan-level
metrics (MAPE on the reference support, wMAPE, support precision/recall), mean and median.

## Tests

```bash
pytest torot/tests
```
