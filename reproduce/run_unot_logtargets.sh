#!/usr/bin/env bash
# UNOT at eps = 0.001 with log-domain Sinkhorn targets (the official float32 scaling-domain targets underflow at this eps
# on our cost scale), widths 1024 and the official 4 n; the runs join the existing UNOT runs and validation chooses
# among all of them per cost. Then: UNOT predictions, UNOT + 10 Sinkhorn iterations (unot_sk10) and, at n = 30, the
# warm-start curves.
#   N=30 CR=experiments/torot/camera_ready OUT=experiments/baselines bash reproduce/run_unot_logtargets.sh [gpu]
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
N=${N:-30}; CR=${CR:-$REPO/experiments/torot/camera_ready}; OUT=${OUT:-$REPO/experiments/baselines}
CONDA_ENVS=${CONDA_ENVS:-$(conda info --base 2>/dev/null)/envs}
PY=${PY_UNOT:-$CONDA_ENVS/env_unot/bin/python}
[ -n "${1:-}" ] && export CUDA_VISIBLE_DEVICES=$1
export PYTHONUNBUFFERED=1
mkdir -p "$OUT/logs"
for cost in sqeuclidean euclidean canberra sqrt_euclidean cubic_euclidean gaussian; do
  for width in 1024 $((4 * N)); do
    run=$OUT/unot/${cost}_eps0.001_logt_w$width
    [ -f "$run/metrics.json" ] && continue
    "$PY" -m baselines.unot.train --cost $cost --eps 0.001 --width $width --log-targets --n $N --out "$run" --val "$CR" \
      > "$OUT/logs/unot_${cost}_eps0.001_logt_w$width.log" 2>&1 &
  done
done
wait
SETS=('id_simplex_*' 'ood_paper_*')
"$PY" -m baselines.unot.predict --root "$CR" --runs "$OUT/unot/*" --sets "${SETS[@]}" 2>&1 | grep -v Warning
UNOT_SINKHORN_STEPS=10 "$PY" -m baselines.unot.predict --root "$CR" --runs "$OUT/unot/*" --sets "${SETS[@]}" 2>&1 | grep -v Warning
[ "$N" = 30 ] && "$PY" -m baselines.unot.refine --root "$CR" --runs "$OUT/unot/*" --sets 'id_simplex_*' 'ood_paper_*' 2>&1 | grep -v Warning
"$PY" -m pip freeze > "$OUT/env_unot.freeze.txt"
echo "$(date +%F_%T) done (N = $N)"
