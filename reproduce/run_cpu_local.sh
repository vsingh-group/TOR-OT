#!/usr/bin/env bash
# CPU part of the experiments (scores and figures), e.g. on a local machine after copying the results
# folder there:
#   rsync -av <server>:<TOR-OT>/experiments/torot/camera_ready/ <TOR-OT>/experiments/torot/camera_ready/
#   conda activate env_torot && pip install -r reproduce/requirements.txt
#   bash reproduce/run_cpu_local.sh experiments/torot/camera_ready
set -euo pipefail
CR=$(cd "${1:?usage: run_cpu_local.sh <camera_ready dir>}" && pwd)
cd "$(dirname "${BASH_SOURCE[0]}")/.."                         # repository root
mkdir -p "$CR/logs"
python reproduce/score.py "$CR" 2>&1 | tee "$CR/logs/score.log"
python reproduce/score_reference.py "$CR" 2>&1 | tee "$CR/logs/score_reference.log"   # Fig. 3: each method vs its own problem
python reproduce/make_figures.py "$CR" 2>&1 | tee "$CR/logs/make_figures.log"
