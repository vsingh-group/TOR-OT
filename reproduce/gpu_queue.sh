#!/usr/bin/env bash
# GPU job queue: one worker per GPU takes "name|command" jobs in order; a worker starts a job only when its GPU runs
# no compute process of ours, so there is one job per GPU at a time and timings are clean.
#   bash reproduce/gpu_queue.sh <jobs file> [gpu ...]          (default GPUs: 0 1)
# The jobs file (bash, see reproduce/jobs/) appends to JOBS and may use REPO, CR, EVAL, BASE, CONDA_ENVS and
# the interpreters PY (env_torot), PY_NEURALOT (env_neuralot), PY_METAOT (env_metaot), PY_UNOT (env_unot). Paths can be overridden via the environment.
# Status: <camera_ready>/queue/<jobs name>/status.tsv; logs: <camera_ready>/logs/queue/<job>.log
# ALLOW_SHARED=1 skips the free-GPU check (jobs still run one at a time per GPU): for jobs whose timings are not used.
# STOP_CMD='<shell test>': checked before each job; when it succeeds the workers take no new job (running ones finish).
# MIN_FREE_MB=<MB>: a worker starts a job only when its GPU has at least this much free memory (shared GPUs).
set -u
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CR=${CR:-$REPO/experiments/torot/camera_ready}
EVAL=${EVAL:-$REPO/experiments/torot/row_M_5M}                 # trained ToR-OT run (best.pt, ckpt_*.pt)
BASE=${BASE:-$REPO/experiments/baselines}                      # trained learned baselines
CONDA_ENVS=${CONDA_ENVS:-$(conda info --base 2>/dev/null)/envs}
PY=${PY:-$CONDA_ENVS/env_torot/bin/python}
PY_NEURALOT=${PY_NEURALOT:-$CONDA_ENVS/env_neuralot/bin/python}
PY_METAOT=${PY_METAOT:-$CONDA_ENVS/env_metaot/bin/python}
PY_UNOT=${PY_UNOT:-$CONDA_ENVS/env_unot/bin/python}
JOBS_FILE=${1:?usage: gpu_queue.sh <jobs file> [gpu ...]}; shift
GPUS=("$@"); [ ${#GPUS[@]} -eq 0 ] && GPUS=(0 1)
JOBS=()
source "$JOBS_FILE"
Q=$CR/queue/$(basename "$JOBS_FILE" .sh)
mkdir -p "$Q" "$CR/logs/queue"

gpu_has_our_process() {   # 0 if GPU $1 runs a compute process owned by us
  local pid
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" 2>/dev/null); do
    [ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" = "$USER" ] && return 0
  done
  return 1
}
next_job() { ( flock 9; i=$(cat "$Q/next"); echo $((i + 1)) > "$Q/next"; echo "$i" ) 9>"$Q/lock"; }
worker() {
  local gpu=$1 i name cmd rc
  while true; do
    [ -z "${ALLOW_SHARED:-}" ] && while gpu_has_our_process "$gpu"; do sleep 30; done   # ALLOW_SHARED=1: run beside other jobs
    [ -n "${MIN_FREE_MB:-}" ] && while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")" -lt "$MIN_FREE_MB" ]; do sleep 60; done
    [ -n "${STOP_CMD:-}" ] && ( cd "$REPO" && bash -c "$STOP_CMD" ) && { printf "%s\tGPU%s\tstopped (STOP_CMD)\n" "$(date +%F_%T)" "$gpu" >> "$Q/status.tsv"; break; }
    i=$(next_job); [ "$i" -ge "${#JOBS[@]}" ] && break
    name=${JOBS[$i]%%|*}; cmd=${JOBS[$i]#*|}
    printf "%s\tGPU%s\tstart\t%s\n" "$(date +%F_%T)" "$gpu" "$name" >> "$Q/status.tsv"
    ( cd "$REPO" && CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" ) > "$CR/logs/queue/$name.log" 2>&1; rc=$?
    printf "%s\tGPU%s\tend rc=%s\t%s\n" "$(date +%F_%T)" "$gpu" "$rc" "$name" >> "$Q/status.tsv"
  done
}
echo 0 > "$Q/next"
printf "%s\tqueue\t%s jobs\t%s\n" "$(date +%F_%T)" "${#JOBS[@]}" "$(printf '%s ' "${JOBS[@]%%|*}")" >> "$Q/status.tsv"
for gpu in "${GPUS[@]}"; do
  worker "$gpu" &
  sleep 5                                 # stagger the workers' first pick
done
wait
echo "finished $(date)" > "$Q/QUEUE_DONE"
