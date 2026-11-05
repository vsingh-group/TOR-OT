# Reproducing the paper

Scripts that produce every number, table and figure of the paper. Results are written to
`experiments/torot/camera_ready/` (1D, n = 30) and its siblings `camera_ready_2d/`, `camera_ready_n100/` and
`camera_ready_n200/` (all git-ignored). Baselines live in [`baselines/`](../baselines/README.md); the ToR-OT model in `torot/` (described in the
[main README](../README.md)).

## Environments

| environment | used for | install |
|---|---|---|
| `env_torot` | ToR-OT, sets, POT and Sinkhorn, scoring, figures | top-level README, then `pip install -r reproduce/requirements.txt` |
| `env_metaot` | Meta OT baseline | `baselines/metaot/requirements.txt` |
| `env_unot` | UNOT baseline | `baselines/unot/requirements.txt` |
| `env_neuralot` | NeuralOT baseline | `baselines/neuralot/requirements.txt` |

`pip freeze` snapshots of the environments used are in [`reproduce/envs/`](envs/).

## Pipeline

Run every command from the repository root. GPU work goes through
`bash reproduce/gpu_queue.sh reproduce/jobs/<file>.sh <gpu ...>`: one worker per GPU runs the jobs of the file in order,
one job per GPU at a time, and a rerun skips the jobs already finished (`ALLOW_SHARED=1` lets jobs whose timings are
not used share a GPU with other work). The job files are the runs made for the paper, in the order below; the last
column names what each step adds. Defaults: results `experiments/torot/camera_ready`, 1D model
`experiments/torot/row_M_5M`, learned baselines `experiments/baselines`.

The released weights can replace a trained model wherever a checkpoint is passed, e.g.
`python reproduce/predict_torot.py experiments/torot/camera_ready ivanjaenm/torot-1d`. Only the
training curve of App. G needs the training run itself (its intermediate checkpoints and log).

### 1D, n = 30

| step | command | env | device | adds |
|---|---|---|---|---|
| 1. 1D model | `torchrun --nproc_per_node 8 -m torot.train --config torot/configs/torot_1d.yaml` | `env_torot` | 8 GPUs | `experiments/torot/row_M_5M/` (`best.pt`, `ckpt_*.pt`, `log.jsonl`) |
| 2. sets | `python reproduce/build_sets.py experiments/torot/camera_ready sqeuclidean euclidean canberra sqrt_euclidean cubic_euclidean gaussian` | `env_torot` | CPU | `sets/` (instances and POT's exact plans), `sets_val/` |
| 3. ToR-OT | `bash reproduce/run_lastrow.sh 0 1` | `env_torot` | GPU | `preds/<set>__v2row_lastrow{,_sym}.npy` (+ timing), `eval_row_M_5M/eval_<split>_lastrow.json` |
| 4. learned baselines | `bash reproduce/run_learned_baselines.sh 0 1` | `env_metaot`, `env_unot` | GPU | `experiments/baselines/{metaot,unot}/`, `preds/<set>__{metaot,unot}.npy` |
| (once) ablation corpus | `python reproduce/build_n30_corpus.py experiments/torot/ablations/corpus_n30` (and `--from-local experiments/torot/ablations/corpus_n30 --train-fraction 0.25` etc., see the script) | `env_torot` | CPU | `experiments/torot/ablations/corpus_n30*` |
| 5. extras | `camera_ready_extra.sh`, then `camera_ready_extra2.sh` and `camera_ready_extra3.sh` (re-runs: NeuralOT with less memory per GPU, Meta OT timing on an idle GPU) | all | GPU | clean timings of Meta OT / UNOT, `warmstart/` (baselines + k Sinkhorn iterations), NeuralOT on 10 instances per set (`preds_subset/`), ablations (seed 0), training seeds 1 and 2 of Meta OT / UNOT |
| 6. latency, ablation seeds | `block_a.sh` | `env_torot` | idle GPU | `latency/latency_row_M_5M_lastrow.json`, ablations (seeds 1 and 2) |
| 7. shared-GPU block | `ALLOW_SHARED=1 ... block_b_shared.sh` | `env_torot`, `env_neuralot` | GPU | `interp/`, n = 100 ToR-OT and learned baselines, `eval_row_M_5M/eval_<split>_sym_lastrow.json`, `data_curve/`, NeuralOT on 100 test instances per cost |
| 8. NeuralOT, every instance | `neuralot_full.sh` (all 144 OOD pairs per cost), `neuralot_test_all.sh` (all 1,000 test instances per cost) | `env_neuralot` | GPU | `preds/<set>__neuralot.npy` |
| 9. interpretability on every set, GPU Sinkhorn, testbed scaling | `extras_interp_all_sets.sh`, `extras_gpu.sh` | `env_torot` | GPU | `interp_sets/`, `preds/<set>__sinkhorn_gpu_eps{0.01,0.001}.npy` (the entropic references; also at n = 100), ablations `corpus10/50`, `d64/384` (3 seeds) |
| 10. UNOT log-domain targets | `unot_logtargets.sh`, then `fill_extra.sh` | `env_unot`, `env_torot` | GPU | UNOT runs with log-domain targets (n = 30, 100, 200; validation picks one run per cost), new UNOT predictions, `preds/<set>__unot_sk10.npy` and the `warmstart/` curves (n = 30), UNOT seeds 1 and 2 at n = 30, Sinkhorn references at n = 200, Meta OT / UNOT seeds at n = 100 and 200 |
| 11. scores and figures | `bash reproduce/run_cpu_local.sh experiments/torot/camera_ready` (`score.py`, `score_reference.py`, `make_figures.py`) | `env_torot` | CPU | `scores/per_instance/`, `scores/per_instance_ref/`, `paper/` |
| 12. model release | `python reproduce/export_to_hub.py --kind 1d --checkpoint experiments/torot/row_M_5M/best.pt --out experiments/hub/torot-1d --repo-id ivanjaenm/torot-1d --baselines experiments/baselines` (2D: `--kind 2d`, see the script), then `hf upload` | `env_torot` | CPU | `experiments/hub/torot-1d/` |

Steps 5–10 are `bash reproduce/gpu_queue.sh reproduce/jobs/<file> <gpus>`; each file's header gives the command used.
Optional: `python -m baselines.pot.solve experiments/torot/camera_ready` adds CPU timings of POT and of POT's Sinkhorn
(rows of the runtime table that the paper does not report).

### n = 100 and n = 200 (App. G)

| step | command |
|---|---|
| sets | `python reproduce/build_sets.py experiments/torot/camera_ready_n100 <the six costs> --n 100` (`--val-only` for `sets_val/` alone); n = 200: `camera_ready_n200`, `--n 200` |
| ToR-OT, Meta OT, UNOT at n = 100 | step 7 (`block_b_shared.sh`) |
| NeuralOT at n = 100 | `neuralot_n100.sh` (200 test instances and the 144 OOD pairs per cost), then `neuralot_n100_full.sh` (all 1,000 test instances) |
| ToR-OT at n = 200 | `ALLOW_SHARED=1 ... n200_torot.sh` |
| Meta OT, UNOT at n = 200 | `N=200 CR=experiments/torot/camera_ready_n200 OUT=experiments/baselines_n200 MAX_PAR=8 bash reproduce/run_learned_baselines.sh 2 3`, then steps 9–10 (Sinkhorn references, log-domain UNOT, seeds) |
| scores | `python reproduce/score.py <root>` and `python reproduce/score_reference.py <root>` for both roots |

### 2D (Fig. 4C/D, Fig. 5, App. F)

Results in `experiments/torot/camera_ready_2d` (6 x 6 grid on [0, 1]², n = m = 36; `make_figures.py` reads them as
`<root>_2d`, or `--root-2d`). The OOD figures use all 7 distributions (49 pairs), 8 Gaussians included
(`ood_2d_exclude: []` in `plot_style.yaml`).

| step | command | env | device |
|---|---|---|---|
| 1. sets | `python reproduce/build_sets_2d.py experiments/torot/camera_ready_2d`: `id_simplex2d_<cost>` (1,000 pairs) and `ood2d_paper_<cost>` (the paper's 2D OOD set: 49 pairs of 7 distributions) | `env_torot` | CPU |
| 2. 2D model | `torchrun --nproc_per_node 2 -m torot.train --config torot/configs/torot_2d.yaml`; the run itself was launched with `DATA=ivanjaenm/torot_data_4M_2d HOURS=64 bash reproduce/run_train_2d.sh 2,3 experiments/torot/row_M_2d_mixed_support` (same recipe; 627,000 steps from the time budget) | `env_torot` | 2 GPUs |
| 3. ToR-OT | `CKPT2D=<2D run>/best.pt bash reproduce/jobs/predict_torot_2d_cpu.sh` (CPU, then scores). On GPUs: `CR=$PWD/experiments/torot/camera_ready_2d CKPT2D=<2D run>/best.pt bash reproduce/gpu_queue.sh reproduce/jobs/predict_torot_2d.sh 2 3` | `env_torot` | CPU / GPU |
| 4. baselines | `DIM=2 N=36 CR=experiments/torot/camera_ready_2d OUT=experiments/baselines_2d bash reproduce/run_learned_baselines.sh 2 3`; `neuralot_2d.sh` and `fill_2d.sh` (NeuralOT on L₂² and L₂, Meta OT / UNOT seeds 1, 2); `python -m baselines.pot.sinkhorn_gpu --root experiments/torot/camera_ready_2d --sets 'id_simplex2d_*' 'ood2d_paper_*'` (references) | all | GPU |
| 5. scores | `python reproduce/score.py experiments/torot/camera_ready_2d` and `python reproduce/score_reference.py experiments/torot/camera_ready_2d --sets 'id_simplex2d_*' 'ood2d_paper_*'` | `env_torot` | CPU |
| 6. figures | step 11 of the 1D pipeline makes the 2D figures too (or `--only fig4_camera_ready fig5_camera_ready fig_appF_distributions fig_appF_cost_summary fig_appF_pairs fig_appF_allcosts fig_appF_improvement`) | `env_torot` | CPU |

## Evaluation sets (n = m = 30)

| set | instances | what |
|---|---|---|
| `ood_paper_<cost>` | 144 | the OOD set: 12 named distributions, all ordered pairs, as histograms of 1,000,000 samples per distribution (torch seed 0) |
| `id_simplex_<cost>` | 1000 | the test set: random instances with the training marginals of the corpus (uniform on the simplex); seeds from 2e9 |
| `sets_val/val_simplex_<cost>` | 500 | validation only (model selection of the learned baselines); seeds from 3e9 |

The OOD histograms come from 1,000,000 samples per distribution; with fewer CPU threads than the machine has (e.g.
`OMP_NUM_THREADS=1`), PyTorch's samplers round differently and one or two samples per histogram can land in a
neighbouring bin (at most 2e-6 of the mass, in about 24 of the 144 pairs). Build the sets with the default thread
count; the other sets do not depend on it.

Costs: the six cost types of the corpus (`torot/data/cost.py`; paper notation in the
[main README](../README.md#cost-types)): `euclidean` (L₂; in 1D equal to L1, L3 and L∞), `sqrt_euclidean` (L₂^½),
`sqeuclidean` (L₂²), `cubic_euclidean` (L₂³), `canberra`, `gaussian`; the marginals are identical across costs. All seeds
are disjoint from each other and from the training corpora (seeds < 1.1e7). Every method is run
on exactly these instances.

## Metrics (`score.py`)

`paper_mape` is the paper's MAPE (weighted MAPE over the union of supports = L1 plan error relative to the exact
plan). Also: MAE, Hellinger, total variation, |OT cost difference|, support recall / precision, row / column / max
marginal violation and the signed cost gap. `euclidean` has non-unique optimal plans in 1D, so plan-matching metrics
there depend on which optimum POT returns: judge it by cost gap and marginal violations.

`score_reference.py` (used by Fig. 3) scores each method against the solution of the problem
it solves: Meta OT and UNOT solve entropic OT, so their reference is the converged Sinkhorn plan at the eps of the
selected run; ToR-OT and NeuralOT target exact OT, so their reference is the optimal plan closest to the prediction
(a linear program over the optimal face; POT's plan whenever the optimum is unique, which removes the `euclidean`
tie problem). Output: `scores/per_instance_ref/<set>__<method>.csv` (MAPE against that reference and against POT).

## Decoding variants of ToR-OT

The paper's ToR-OT decodes every row of the plan with the model, the last one included (`v2row_lastrow` in file
names; `torot.decode.decode_by_rows(last_row="model")`, `--last_row model` in `torot.evaluate`, the default of
`predict_torot.py`). The alternative `last_row="remainder"` sets the last row to the remaining column capacity: it keeps
the column marginals exact up to over-filled columns, but collects every small error of the earlier rows as a
scattering of tiny entries in the last row, which inflates the transport cost. `_sym` variants average the plans of
(a, b) and (b, a) (transpose averaging, App. C.2).

## Ablations

Small testbed (`torot/configs/ablation_n30.yaml`: row mode, 9.8M parameters, n <= 30 subset of `torot_data_1M`,
11,000 steps on one GPU). `reproduce/run_ablation.sh <name>` changes one entry of the base recipe (transpose
augmentation, loss, capacity perturbation, model size, corpus size; `base_seed1` measures the seed spread), trains
and evaluates `test_id` / `test_ood` of that corpus. Runs: `experiments/torot/ablations/<name>/`.

## Figures and tables of the paper

`python reproduce/make_figures.py experiments/torot/camera_ready` writes every data-driven figure and table of the
paper, and nothing else, into `experiments/torot/camera_ready/paper/`, one folder per place in the paper. Figures are
PDF and PNG (App. E and F: PNG; App. G: PDF), tables Markdown, LaTeX and CSV; the numbers drawn in each figure are in
`<folder>/data/` (CSV), to re-plot them with any tool. Folders and file names are set in one place, `ROUTES` at the
top of `make_figures.py`. A function whose inputs are missing is skipped and reported at the end.

| paper | output in `paper/` | function (`--only`) | inputs (pipeline step) |
|---|---|---|---|
| Fig. 1, Fig. 2 | — | — | drawings (Fig. 2: TikZ in the paper source), not made from results |
| Fig. 3 | `main_text/fig3_evaluation_mape` (`_otcost_diff`: the same with the OT-cost difference) | `fig3_camera_ready` | `scores/per_instance_ref/` (11) of ToR-OT (3), NeuralOT (8), Meta OT / UNOT and their seeds (4, 5, 10), against the Sinkhorn references (9) |
| Fig. 4 | `main_text/fig4_results` | `fig4_camera_ready` | A, B: 1D scores (3, 4, 11); C, D: 2D scores (2D 3–5) |
| Table 1 | `main_text/table1_improvement` | `table1_improvement` | `scores/per_instance/` (11) of ToR-OT, NeuralOT, Meta OT, UNOT |
| Fig. 5 | `main_text/fig5_2d_example` | `fig5_camera_ready` | 2D predictions and scores (2D 3–5) |
| App. A | — | — | no figures; the hyperparameters are `torot/configs/torot_1d.yaml` and `torot_2d.yaml` |
| App. B runtime table | `appendix_B_runtime/table_appB_runtime` | `fig6_and_table2_runtime` | timing files `preds/<set>__<method>.timing.json` (3, 5, 8, 9, 10) |
| App. B amortized runtime | `appendix_B_runtime/fig_appB_amortized_runtime` | `fig6_and_table2_runtime` | ToR-OT timing (3); `--train-hours`, `--neuralot-seconds` |
| App. B latency vs n | `appendix_B_runtime/fig_appB_latency_vs_n` | `fig_latency_vs_n` | `latency/` (6) |
| App. C.1 Sinkhorn projection | `appendix_C_feasibility/table_appC_sinkhorn_projection` | `table_appC_sinkhorn` | ToR-OT predictions (3) |
| App. C.1 Sinkhorn refinement | `appendix_C_feasibility/fig_appC_sinkhorn_refinement_{iters,time}` | `fig_warmstart` | `warmstart/` (5, 10) |
| App. C.2 transpose averaging | `appendix_C_feasibility/table_appC_transpose_averaging_{sets,5M_splits}` | `table4_symmetric`, `table4_symmetric_5M` | scores of `v2row_lastrow` and `v2row_sym_lastrow` (3, 11); `eval_row_M_5M/` (3, 7) |
| App. D | `appendix_D_interpretability/fig_appD_interpretability`, `table_appD_interpretability_all_sets` | `fig_appD_interpretability`, `table_appD_all_sets` | `interp/` (7), `interp_sets/` (9) |
| App. E | `appendix_E_1d_results/fig_appE_*` (12 OOD distributions, per-cost summaries, plans of the 144 OOD pairs per method, all costs, improvement over Meta OT) | `fig_appE_distributions`, `fig_appE_cost_summary`, `fig_appE_pairs`, `fig_appE_allcosts`, `fig_appE_improvement` | 1D sets, predictions and scores (2–11) |
| App. F | `appendix_F_2d_results/fig_appF_*` (the same on the 2D sets) | `fig_appF_*` | 2D pipeline |
| App. G support size | `appendix_G_scaling/table_appG_support_size` | `table_appG_support_size` | scores at n = 30 and 100, `latency/` |
| App. G scaling to n = 100 | `appendix_G_scaling/fig_appG_scaling_n100`, `table_appG_scaling_n100` | `fig_scaling_n100` | scores at n = 30, 100 and 200 |
| App. G error vs support size | `appendix_G_scaling/fig_appG_size_generalization` | `fig_size_generalization` | `eval_row_M_5M/` (3) |
| App. G training curve | `appendix_G_scaling/fig_appG_training_curve` | `fig_appG_training_curve` | `data_curve/` (7), the training log `log.jsonl` (1; `--train-run`) |
| App. G testbed, ablations | `appendix_G_scaling/fig_appG_testbed`, `table_appG_ablations` | `fig_appG_testbed`, `table_ablations` | `experiments/torot/ablations/<run>/` (5, 6, 9) |

* **Style**: labels, colors, colormaps, fonts, file formats and resolution are in `reproduce/plot_style.yaml`; edit it
  (or pass `--style <copy>`) and rerun `make_figures.py`.
* **One figure at a time**: `python reproduce/make_figures.py experiments/torot/camera_ready --only fig4_camera_ready
  table1_improvement`.
* **Other folders**: `--out` (default `<results>/paper`), `--root-2d` (default `<results>_2d`), `--n100-root` (default `<results>_n100`; n = 200 is read from
  `<results>_n200`), `--ablations` (default `<results>/../ablations`), `--train-run` (default
  `experiments/torot/row_M_5M`).
