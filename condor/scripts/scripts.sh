#!/bin/bash

#rm -rf condor_log
#rm logs/condor_logs/*

get_quotas /staging/jaenmarquez

# Debug / Test
condor_submit -i experiment=dataset-interactive  condor/scripts/cpu-job.sub
condor_submit -i experiment=training-interactive condor/scripts/gpu-job.sub

# Run a single experiment
condor_submit experiment=dataset  condor/scripts/cpu-job.sub
condor_submit experiment=exp1 condor/scripts/gpu-job.sub