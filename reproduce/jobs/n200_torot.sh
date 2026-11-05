# ToR-OT at n = 200 (trained on n <= 100: extrapolation), test and paper-OOD sets, one job per cost, beside the n = 200
# training of the learned baselines (timings of this run are not used; latency vs n is measured by torot.benchmark):
#   ALLOW_SHARED=1 bash reproduce/gpu_queue.sh reproduce/jobs/n200_torot.sh 2 3
X="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1"
CR200=${CR200:-$REPO/experiments/torot/camera_ready_n200}
for cost in sqeuclidean euclidean canberra sqrt_euclidean cubic_euclidean gaussian; do
  JOBS+=("n200_torot_$cost|$X $PY $REPO/reproduce/predict_torot.py $CR200 $EVAL/best.pt --last-row model --plain-only id_simplex_$cost ood_paper_$cost")
done
