# Block B/C, jobs that run beside NeuralOT (reproduce/jobs/neuralot_full.sh) on the same GPUs; their timings are not
# used. One job at a time per GPU:
#   ALLOW_SHARED=1 bash reproduce/gpu_queue.sh reproduce/jobs/block_b_shared.sh 2 3
X="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1"
CR100=${CR100:-$REPO/experiments/torot/camera_ready_n100}
# App. D: interpretability of the row model (paper's OOD set, sqeuclidean)
JOBS+=("interpretability|$X $PY $REPO/reproduce/interpretability.py $CR $EVAL/best.pt --set ood_paper_sqeuclidean")
# scaling with the number of bins: n = 100 (ToR-OT; Meta OT and UNOT trained at n = 100)
JOBS+=("n100_torot|$X $PY $REPO/reproduce/predict_torot.py $CR100 $EVAL/best.pt --last-row model --plain-only 'id_simplex_*' 'ood_paper_*'")
JOBS+=("n100_learned_baselines|N=100 CR=$CR100 OUT=$REPO/experiments/baselines_n100 MAX_PAR=8 bash $REPO/reproduce/run_learned_baselines.sh \$CUDA_VISIBLE_DEVICES")
# Table 4 and App. E with the last row decoded by the model
mkdir -p "$CR/eval_row_M_5M" "$CR/data_curve"
for split in test_ood test_id test_extrap; do
  extra=""; [ "$split" = test_extrap ] && extra="--max_instances 1000"
  JOBS+=("sym_lastrow_$split|$X $PY -m torot.evaluate --checkpoint $EVAL/best.pt --split $split --symmetric --last_row model --batch_size 64 $extra --out $CR/eval_row_M_5M/eval_${split}_sym_lastrow.json")
done
for step in 025000 050000 075000 100000 125000 150000 175000 200000; do
  JOBS+=("data_curve_lastrow_step$step|$X $PY -m torot.evaluate --checkpoint $EVAL/ckpt_$step.pt --split test_id --max_instances 1000 --batch_size 64 --last_row model --out $CR/data_curve/step${step}_test_id_lastrow.json && $X $PY -m torot.evaluate --checkpoint $EVAL/ckpt_$step.pt --split test_ood --max_instances 864 --batch_size 64 --last_row model --out $CR/data_curve/step${step}_test_ood_lastrow.json")
done
# Block C: NeuralOT on 100 test instances per cost
JOBS+=("neuralot_test_a|$X $PY_NEURALOT -m baselines.neuralot --root $CR --sets id_simplex_sqeuclidean id_simplex_euclidean id_simplex_canberra --per-set 100 --workers 8")
JOBS+=("neuralot_test_b|$X $PY_NEURALOT -m baselines.neuralot --root $CR --sets id_simplex_sqrt_euclidean id_simplex_cubic_euclidean id_simplex_gaussian --per-set 100 --workers 8")
