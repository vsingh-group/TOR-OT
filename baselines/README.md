# Baselines

Every method ToR-OT is compared against in the paper. Each baseline writes predictions in one
format: dense plans `preds/<set>__<method>.npy` plus `preds/<set>__<method>.timing.json`, for the evaluation sets
built by [`reproduce/build_sets.py`](../reproduce/build_sets.py). Scoring and figures are done by
[`reproduce/`](../reproduce/README.md).

| method | what it is | code | environment | device |
|---|---|---|---|---|
| `pot` | exact network simplex (`ot.emd`): the reference plans, stored in the sets by `reproduce/build_sets.py`; `pot/solve.py` adds its CPU timing (optional, not in the paper) | [`pot/solve.py`](pot/solve.py) | `env_torot` | CPU |
| `sinkhorn_eps*` | entropic OT, log-domain Sinkhorn (POT) on CPU, run to convergence (optional, not in the paper) | [`pot/solve.py`](pot/solve.py) | `env_torot` | CPU |
| `sinkhorn_gpu_eps*` | log-domain Sinkhorn on GPU (PyTorch), batched, with clean timings; the reference plans of the entropic methods | [`pot/sinkhorn_gpu.py`](pot/sinkhorn_gpu.py) | `env_torot` | GPU |
| `metaot` | Meta OT (Amos et al., ICML 2023): MLP predicts the entropic dual potential | [`metaot/`](metaot/) (official code, vendored) | `env_metaot` | GPU |
| `unot` | UNOT (Geuter et al., ICML 2025): network predicts Sinkhorn scalings, adversarial generator | [`unot/`](unot/) (official code, vendored) | `env_unot` | GPU |
| `neuralot` | NeuralOT (Korotin et al., ICLR 2023), trained from scratch for every instance; every instance at n = 30 and n = 100; in 2D the costs L₂² (every instance) and L₂ (200 test instances) | [`neuralot/`](neuralot/) (official code, vendored) | `env_neuralot` | GPU |

## Environments

Every result is produced inside a pinned environment; `pip freeze` snapshots are saved next to the results.

```bash
conda create -y -n env_metaot python=3.10 && conda run -n env_metaot pip install -r baselines/metaot/requirements.txt
conda create -y -n env_unot  python=3.12 && conda run -n env_unot  pip install -r baselines/unot/requirements.txt
conda create -y -n env_neuralot python=3.11 && conda run -n env_neuralot pip install -r baselines/neuralot/requirements.txt
```
`env_torot` is the ToR-OT environment (top-level README); exact versions of every environment: [`reproduce/envs/`](../reproduce/envs/).
Run every baseline from the repository root (`python -m baselines....`): they import `torot.data.cost` (numpy + POT)
from the checkout, so the cost matrices are exactly those of the evaluation sets.

## Learned baselines: setting and protocol

* **Problems**: pairs of 30-bin histograms on the grid of bin centres in [0, 1], max-normalised cost (C.max() = 1),
  for the six cost types of ToR-OT's corpus (`torot/data/cost.py`): `sqeuclidean`, `euclidean` (= L1 = L∞ in 1D),
  `canberra`, `sqrt_euclidean`, `cubic_euclidean`, `gaussian`; one model per cost and regularisation, as both
  methods assume a fixed cost.
* **Training data**: Meta OT samples histogram pairs uniformly from the simplex, ToR-OT's training marginals.
  UNOT generates its own training measures with its adversarial generator (no dataset), as in the paper.
* **Entropic regularisation**: both methods solve entropic OT; each is trained with eps = 0.01 (their papers'
  default) and eps = 0.001 (closer to exact OT). The run with the lower MAPE on the validation set
  `sets_val/val_simplex_<cost>` (500 instances, never used for reporting) is used for every evaluation set; the
  choice is recorded in each timing file.
* **UNOT specifics** (all in `unot/train.py`): hidden width 1024 (as Meta OT) and the official rule 4 x n = 120 are
  both trained, and validation chooses together with eps. After training, the predictor's BatchNorm running
  statistics are re-estimated on the trained generator's finite samples (`recalibrate_batchnorm`): the official loop
  updates them with every generated batch, and for 30-bin histograms the generator occasionally emits an all-zero
  measure that its normalisation turns into NaN, which poisons the running statistics (weights are unaffected; the
  pre-recalibration predictor is kept as `predictor_trained.pt`). With eps = 0.001 the official float32 Sinkhorn
  targets underflow on this cost scale (most training losses are non-finite; counts in each `metrics.json`).
* **Predicted plan**: the network's output turned into a plan exactly as the method does before (optional) Sinkhorn
  refinement, with zero Sinkhorn iterations: Meta OT `exp((f + g(f) - C)/eps)` with g the c-transform of the
  predicted f (column marginals exact); UNOT `diag(a / Kv) K diag(v)` with v the predicted scaling (row marginals
  exact).
* **Official code, unchanged**: `metaot/third_party/` and `unot/third_party/` are verbatim copies at pinned commits
  (see their `SOURCE.md`); `neuralot/third_party/` is the official NeuralOT code plus our scripts adapted from its 1D and 2D notebooks (its `SOURCE.md` lists the differences). The adapters `metaot/train.py` and `unot/train.py` only supply what the setting needs
  (input size, cost matrix, sampler, a 1D skip connection for UNOT's generator, network width) and document every
  choice in their docstrings.

```bash
bash reproduce/run_learned_baselines.sh 0 1      # train all runs on GPUs 0 and 1, then predict
# or a single run / prediction:
python -m baselines.metaot.train --cost sqeuclidean --eps 0.01 --out experiments/baselines/metaot/sqeuclidean_eps0.01 \
    --val experiments/torot/camera_ready
python -m baselines.metaot.predict --root experiments/torot/camera_ready --runs 'experiments/baselines/metaot/*' \
    --sets 'id_*' 'ood_paper_*'
```

## Tests

```bash
pytest baselines/tests/test_common.py                   # any environment
conda run -n env_metaot pytest baselines/tests/test_metaot.py
conda run -n env_unot  pytest baselines/tests/test_unot.py
```
The adapter tests run the official training loops for a few iterations and check the plans' marginals.

## Learned initialisation + Sinkhorn

`python -m baselines.{metaot,unot}.refine ...` runs each method's prediction followed by k = 0..1000 Sinkhorn
iterations (Meta OT: ott-jax potential updates; UNOT: its scaling updates in the log domain), and Sinkhorn from its
standard start at the same eps, recording MAPE and time per k in `<camera_ready>/warmstart/`. k = 0 is the plan of
`predict.py`.

## Seeds

`bash reproduce/run_baseline_seeds.sh <metaot|unot> <seed>` retrains the setting chosen on validation (per cost) with
another seed and predicts the sets as `<method>_seed<seed>`; the figures show the mean and standard deviation over
seeds.

## Licenses

The vendored official code keeps its own license: Meta OT is **CC BY-NC 4.0** (non-commercial use only,
`metaot/third_party/LICENSE`); UNOT and NeuralOT are MIT (`unot/third_party/LICENSE`, `neuralot/third_party/LICENSE`). The rest of this folder is under the
repository's license.
