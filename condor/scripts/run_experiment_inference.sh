#!/bin/bash

# experiment_name=$1
# staging_model_directory=/staging/jaenmarquez/models
# num_gpu_devices=$3
# cluster_id=$4

experiment_name="inference"
staging_model_directory="/staging/jaenmarquez/models"
num_gpu_devices="1"
cluster_id="0000"

# old parameters
# #num_workers=$((num_gpu_devices*4))
# num_workers=0
# batch_size_per_device=64
# batch_size=$((num_gpu_devices*batch_size_per_device))

# printing values
# echo "experiment_name: $experiment_name"
# echo "data directory:  $data_directory"
# echo "experiment_id:   $experiment_id"
# echo "num_gpu_devices: $num_gpu_devices"
# echo "num_workers:     $num_workers"
# echo "batch_size:      $batch_size"
# echo "batch_size_per_device: $batch_size_per_device"

echo "checking nvidia-smi:"
nvidia-smi
#echo "checking nvidia-smi nvlink:"
#nvidia-smi nvlink --status
#echo "checking nvcc version:"
#nvcc --version
#echo "checking shared memory (/dev/shm):"
#df -h
#echo "running system check:"
#/usr/bin/python3 src/utils/system_check.py

#export NETRC=/staging/jaenmarquez/.config/wandb/.netrc
#wandb login --relogin ***REMOVED***

export WANDB_API_KEY=***REMOVED***
export HF_HOME=$staging_model_directory
export HF_TOKEN="***REMOVED***"
export TRITON_CACHE_DIR="/staging/jaenmarquez/.cache/triton_cache"
export HF_DATASETS_CACHE="/staging/jaenmarquez/.cache/huggingface/dataset"

# define the model, dataset and script to run
model_folder_name="Llama-3.2-1B"
dataset_folder_name=hierarchical_QA_generic

# Evaluate Ultrametric DPO
script_name=evaluator/evaluator.py
experiment_dir_output=./$model_folder_name-UltraDPO_$cluster_id
# Standard DPO
#script_name=trainer/dpo.py
#experiment_dir_output=./$model_folder_name-DPO_$cluster_id

# copy model weights from /staging to local workspace
echo "copying model files:"
model_dir_orig=$staging_model_directory/$model_folder_name
model_dir_dest=./models/$model_folder_name
echo "  from: $model_dir_orig"
ls -l $model_dir_orig
echo "  to:   $model_dir_dest"
mkdir -p $model_dir_dest
#rsync -ah --progress /staging/jaenmarquez/data/imagenet/ILSVRC2012* ./models
cp -r $model_dir_orig/. $model_dir_dest
ls -l $model_dir_dest

# copy dataset files from /staging to local workspace
echo "copying dataset files:"
staging_dataset_dir=/staging/jaenmarquez/datasets
dataset_dir_orig=$staging_dataset_dir/$dataset_folder_name/data
dataset_dir_dest=./datasets
echo "  from: $dataset_dir_orig"
ls -l $dataset_dir_orig
echo "  to:   $dataset_dir_dest"
mkdir -p $dataset_dir_dest
cp -r $dataset_dir_orig/. $dataset_dir_dest
ls -l $dataset_dir_dest
#mkdir -p $experiment_dir_output

# run distributed inference job
echo "running inference job:"
accelerate launch \
    --mixed_precision bf16 \
    $script_name \
    --model_path $model_dir_dest \
    --batch_size 256 \
    --pad_to_multiple_of 8 \
    --temperature 0.8

# # copy results back to /staging
# echo "copying results back to /staging:"
# echo "  from: $experiment_dir_output"
# ls -l $experiment_dir_output
# echo "  to:   $staging_model_directory/$experiment_dir_output"
# cp -r $experiment_dir_output/. $staging_model_directory/$experiment_dir_output
# ls -l $staging_model_directory/$experiment_dir_output