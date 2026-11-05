# ToR-OT predictions on the 2D sets (Figure 4 C/D, Figure 5, Appendix F) on GPUs, with the last row decoded by the
# model, as in 1D. Default checkpoint: the 2D run of torot/configs/torot_2d.yaml (CKPT2D to change it); on CPU:
# predict_torot_2d_cpu.sh. One job per cost (its test and OOD sets); the GPUs run nothing else, so the timings are clean:
#   CR=$PWD/experiments/torot/camera_ready_2d bash reproduce/gpu_queue.sh reproduce/jobs/predict_torot_2d.sh 2 3
# Then (CPU): score.py and score_reference.py --sets 'id_simplex2d_*' 'ood2d_paper_*' on experiments/torot/camera_ready_2d.
X="PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
CR2D=${CR2D:-$REPO/experiments/torot/camera_ready_2d}
CKPT2D=${CKPT2D:-$REPO/experiments/torot/row_M_2d_mixed_support/best.pt}
for cost in sqeuclidean euclidean sqrt_euclidean cubic_euclidean canberra gaussian; do
  JOBS+=("torot_2d_$cost|$X $PY reproduce/predict_torot.py $CR2D $CKPT2D id_simplex2d_$cost ood2d_paper_$cost --last-row model")
done
