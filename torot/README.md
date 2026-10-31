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
| Training | Adam, constant LR, fp32 | AdamW (lr 1e-4), warmup + cosine, grad clipping, bf16, EMA, best-val checkpoint; QK-norm in all attention and the pointer head, z-loss |
| Data | one fixed `n` per dataset, rows stored independently; marginals = normalised i.i.d. U(0,1) weights | one HF corpus with `n` ∈ [5, 100] and whole plans per record; marginals uniform on the probability simplex (Dirichlet(1)) by default |

## Decoding modes (`model.decoding`)

* `plan` (default): one sequence per plan, `BOS (ROW (⟨tgt j⟩ E-k d1 d2 d3 d4)*)ⁿ EOS`.
  Row i attends to all tokens of rows 0…i−1. Decoder self-attention costs O((n+m)²).
* `row`: one sequence per row, `BOS (⟨tgt j⟩ E-k d1 d2 d3 d4)* EOS`, generated from the
  *remaining problem*: source bins i…n−1 (bin i, the row being generated, carries a learned
  marker) and the remaining column capacities r = b − Σ_{i'<i} π_{i'} as target masses. After
  each row the predicted masses are subtracted from r; the last row is set to the final r.
  This conditions on the earlier rows without keeping them in context: the rows i…n−1 of an
  optimal plan are an optimal plan of the remaining problem, so row i is the first row of that
  problem's plan (`tests/test_row_decoding.py` checks this with POT). Decoder attention costs
  Σᵢ kᵢ² for kᵢ nonzeros in row i. Each row re-runs the encoder on the remaining problem.
  Training uses one example per (instance, row) built on the fly from the same corpus.
  At test time the capacities come from the model's own (slightly wrong) rows, so a used-up
  column can show a tiny leftover; `data.residual_leftover_prob` / `residual_leftover_range`
  add such leftovers to training examples (target unchanged) so the model learns to ignore
  them, and `data.residual_noise` perturbs the other capacities.

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
python -m torot.data.generate --out_dir ./torot_data_1M --repo_id ivanjaenm/torot_data_1M --push \
    --train 1000000 --val 2500 --test_id 5000 --test_extrap 5000 --seed 0 --workers 16
```

Instances: 1D regular grid on [0, 1], six cost types, exact plan from POT's network
simplex. Marginals are drawn **uniformly from the probability simplex**, i.e.
Dirichlet(1, …, 1) (`--marginals simplex`, the default; no parametric family). v1 and the
corpus `ivanjaenm/torot_data_1M_v2` instead normalised i.i.d. U(0, 1) weights, which is *not*
uniform on the simplex: those histograms stay close to flat (the largest bin averages ~2/n and
a bin essentially never exceeds 3/n, while under Dirichlet(1) about 5 % of bins exceed 3/n at
n = 30). `--marginals normalized_uniform`
reproduces that corpus exactly (its records are labelled `uniform_random`). The cost set differs from v1, where L1, L2, L∞ and Minkowski all collapse
to |x − y| on a line (four of six "types" were the same problem). v2 uses six costs that are
pairwise distinct in 1D (`torot/data/cost.py`; paper notation in parentheses, table in the
[main README](../README.md#cost-types)): `euclidean` |x−y| (L₂; non-unique optima), `sqrt_euclidean` |x−y|^½
(L₂^½; concave), `sqeuclidean` |x−y|² (L₂²), `cubic_euclidean` |x−y|³ (L₂³), `canberra` (Canberra) and
`gaussian` 1 − exp(−|x−y|²/2σ²) (Gaussian). The loader checks the corpus' cost names against this
list and refuses a corpus generated with another cost set. The only new axis is the support size, `n` ∈ [5, 100] per instance
(`n = m`; `equal_sizes: false` in `GenConfig` samples them independently). One record
holds a whole plan, not a row.

Splits: `train`, `val`, `test_id` (same protocol, disjoint seeds), `test_extrap`
(`n` ∈ [120, 200], for extrapolation over the number of bins), and `test_ood` — the
evaluation-only split of 144 ordered pairs of the 12 named 1D densities used for the
paper's OOD evaluation, at 30 and 100 bins and all cost types (`--no_ood` skips it).
Every instance is reproducible from its `seed`.

## Training

The configs default to the corpus `ivanjaenm/torot_data_1M` (marginals uniform on the
simplex, six distinct costs). `ivanjaenm/torot_data_1M_v2` is the earlier corpus with
`normalized_uniform` marginals. `data.repo_id` may also be a local `save_to_disk` directory
produced by `torot.data.generate --out_dir`.

```bash
python -m torot.train --config torot/configs/tiny.yaml                       # sanity run
torchrun --nproc_per_node 4 -m torot.train --config torot/configs/default.yaml
```

Overrides use dot syntax, e.g. `optim.lr=2e-4 run.out_dir=experiments/torot/lr2e-4`.
Checkpoints (`best.pt`, `last.pt`) contain raw and EMA weights; `config.yaml` is saved beside them.

For long runs, launch detached so the job survives a dropped SSH/VPN session:

```bash
torot/run_train.sh 0,1,6,7 experiments/torot/base_25ep optim.max_steps=72000
```

### Monitoring

| | |
|---|---|
| `cat <out_dir>/progress.txt` | one-glance status: step, ETA, latest train/validation metrics, best validation loss (updated every 50 steps) |
| `python -m torot.progress <out_dir>` | table of all validations (`--last N` for the most recent) |
| `tail -f <out_dir>/train.log` | raw stdout/stderr of the run (`run_train.sh` launches) |
| `<out_dir>/log.jsonl` | machine-readable log: one JSON line per 50 training steps and per validation |
| `kill $(cat <out_dir>/train.pid)` | stop a run started with `run_train.sh`; resume with `run.resume=<out_dir>/last.pt` |

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
