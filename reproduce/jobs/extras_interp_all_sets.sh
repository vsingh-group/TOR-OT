# Interpretability (reproduce/interpretability.py) on every other evaluation set: all costs, test and OOD sets.
#   bash reproduce/gpu_queue.sh reproduce/jobs/extras_interp_all_sets.sh 2
X="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1"
for prefix in ood_paper id_simplex; do
  for cost in sqeuclidean euclidean canberra sqrt_euclidean cubic_euclidean gaussian; do
    [ "$prefix$cost" = ood_papersqeuclidean ] && continue                 # the main run (<root>/interp)
    JOBS+=("interp_${prefix}_$cost|$X $PY $REPO/reproduce/interpretability.py $CR $EVAL/best.pt --set ${prefix}_$cost --out $CR/interp_sets/${prefix}_$cost")
  done
done
