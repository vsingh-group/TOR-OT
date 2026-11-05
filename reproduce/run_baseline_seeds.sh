#!/usr/bin/env bash
# Extra training seeds of a learned baseline at the setting chosen on validation (per cost), then predictions.
#   bash reproduce/run_baseline_seeds.sh <metaot|unot> <seed> [gpu]
# Runs: experiments/baselines/seeds/<method>/seed<seed>/<cost>_eps<eps>[_logt][_w<width>]/ (the seed-0 setting: eps, width,
# UNOT's Sinkhorn targets); predictions <set>__<method>_seed<seed> from exactly these runs.
# 2D (reproduce/build_sets_2d.py): DIM=2 N=36 CR=experiments/torot/camera_ready_2d OUT=experiments/baselines_2d
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
METHOD=${1:?method}; SEED=${2:?seed}; GPU=${3:-}
CR=${CR:-$REPO/experiments/torot/camera_ready}
OUT=${OUT:-$REPO/experiments/baselines}
CONDA_ENVS=${CONDA_ENVS:-$(conda info --base 2>/dev/null)/envs}
PY=${PY:-$CONDA_ENVS/env_$METHOD/bin/python}
N=${N:-30}; DIM=${DIM:-1}                                   # as in run_learned_baselines.sh
MAX_PAR=${MAX_PAR:-0}                                       # > 0: at most this many training runs at once (shared GPU)
SETS=('id_simplex_*' 'ood_paper_*'); [ "$DIM" = 2 ] && SETS=('id_simplex2d_*' 'ood2d_paper_*')
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES=$GPU          # otherwise keep the caller's (e.g. the GPU queue)
DEST=$OUT/seeds/$METHOD/seed$SEED; mkdir -p "$DEST" "$OUT/logs"
# the selected seed-0 run per cost: "<cost> <eps> <width or -> <logt or -> <run>" (logt: UNOT's log-domain Sinkhorn targets)
"$PY" - "$OUT/$METHOD" > "$DEST/selected.txt" <<'PYEOF'
import glob, json, sys
from baselines.common import select_run
from torot.data.cost import COST_TYPES
runs = sorted(glob.glob(f"{sys.argv[1]}/*"))
for cost in COST_TYPES:
    run, _ = select_run(runs, cost)
    cfg = json.load(open(f"{run}/config.json"))
    logt = str(cfg.get("sinkhorn_targets", "")).startswith("log domain")
    print(cost, cfg["eps"], cfg.get("hyperparameters", {}).get("width", "-"), "logt" if logt else "-", run)
PYEOF
RUNS=()
while read -r cost eps width logt _; do
  [ "$width" = - ] && width=""; [ "$logt" = - ] && logt=""
  name=${cost}_eps$eps${logt:+_logt}${width:+_w$width}; RUNS+=("$DEST/$name")
  [ -f "$DEST/$name/metrics.json" ] && continue
  [ "$MAX_PAR" -gt 0 ] && while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do wait -n; done
  "$PY" -m baselines.$METHOD.train --cost "$cost" --eps "$eps" --seed "$SEED" ${width:+--width $width} ${logt:+--log-targets} \
    --n "$N" --dim "$DIM" --out "$DEST/$name" --val "$CR" > "$OUT/logs/seed${SEED}_${METHOD}_$name.log" 2>&1 &
done < "$DEST/selected.txt"
wait
"$PY" -m baselines.$METHOD.predict --root "$CR" --runs "${RUNS[@]}" --sets "${SETS[@]}" \
  --method "${METHOD}_seed$SEED" 2>&1 | grep -v "^[WEI]0000\|cuda_\|absl::\|hydra\|compatab\|assume defaults\|version_base\|Warning"
