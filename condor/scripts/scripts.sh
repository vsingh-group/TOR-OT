#!/bin/bash

#rm -rf condor_log
#rm logs/condor_logs/*
username=$(whoami)
get_quotas /staging/$username

# Debug / Test
condor_submit -i experiment=dataset-interactive  condor/scripts/cpu-job.sub
condor_submit -i experiment=TF_100K_cost_semivariable condor/scripts/gpu-job.sub

# Run a single experiment
#condor_submit experiment=dataset  condor/scripts/cpu-job.sub
condor_submit experiment=TF_100K_cost_semivariable condor/scripts/gpu-job.sub
condor_submit experiment=TF_100K_cost_fixed condor/scripts/gpu-job.sub