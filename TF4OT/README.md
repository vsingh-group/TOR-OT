# TF4OT (legacy v1): Transformers for Optimal Transport

> This is the v1 code used for the experiments in the submitted version of the paper.
> The camera-ready implementation lives in [`../torot/`](../torot/). Everything below is
> kept for reproducibility of the original figures. All commands run from the repository root.
>
> Layout: `scripts/` (plotting / patching / inspection helpers), `assets/` (figures, slides),
> `condor/` (HTCondor job files), `evals/` (notebooks and NeuralOT baseline),
> `configs/` (Hydra configs), `requirements.txt` (pinned v1 dependencies, incl. `flash-attn`).

## ⚙️ Python Environment Setup

### Setup via conda environment
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda create --name env_TF4OT python=3.11
conda activate env_TF4OT
pip3 install -e . --no-deps
pip3 install -r TF4OT/requirements.txt
```

### Setup via docker
```bash
docker build -f TF4OT/image.dockerfile -t ivanjaenm/tf4ot:v1 . --push
```

## 🛢️ Synthetic data generator

### Run data generator (cpu)
```bash
python3 -m TF4OT.run_datagenerator --n_bins=10 --metric='sqeuclidean' --dataset_size=100_000 --upload --cost_fixed --row_wise --output_dir=./_otdata_100K_rowwise_costfixed --repo_name='ivanjaenm/otdata_100K_rowwise_cost_fixed'
python3 -m TF4OT.run_datagenerator --n_bins=10 --metric='sqeuclidean' --dataset_size=100_000 --upload --cost_fixed --output_dir=./_otdata_100K_costfixed --repo_name='ivanjaenm/otdata_100K_cost_fixed' 
python3 -m TF4OT.run_datagenerator --n_bins=10 --metric='sqeuclidean' --dataset_size=100_000 --upload --cost_variable --row_wise --output_dir=./_otdata_100K_rowwise_costvariable --repo_name='ivanjaenm/otdata_100K_rowwise_cost_variable'
python3 -m TF4OT.run_datagenerator --n_bins=10 --metric='sqeuclidean' --dataset_size=100_000 --upload --cost_variable --output_dir=./_otdata_100K_costvariable --repo_name='ivanjaenm/otdata_100K_cost_variable'
python3 -m TF4OT.run_datagenerator --n_bins=10 --dataset_size=100_000 --upload --cost_semivariable --row_wise --output_dir=./_otdata_100K_rowwise_costsemivariable --repo_name='ivanjaenm/otdata_100K_rowwise_cost_semivariable'
python3 -m TF4OT.run_datagenerator --n_bins=10 --dataset_size=100_000 --upload --cost_semivariable --output_dir=./_otdata_100K_costsemivariable --repo_name='ivanjaenm/otdata_100K_cost_semivariable'
```
Metric options: ['euclidean', 'sqeuclidean', 'cityblock', 'hamming', 'rogerstanimoto', 'russellrao', 'canberra', 'jaccard']


## 🚀 Training Pipeline Overview

### Run training (gpu)
```bash
CUDA_VISIBLE_DEVICES=7 python3 -m TF4OT.run_training --config-name TF_100K_cost_fixed
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m TF4OT.run_training --config-name TF_100K_cost_fixed
```

## 📊 Evaluation

### Run inference (gpu)
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m TF4OT.run_inference --config-name TF_100K_cost_semivariable
```

### Run tests
```bash
python3 -m TF4OT.tokenizers_test
python3 -m TF4OT.vocabs_test
```

### Run all experiments in the background
```bash
nohup ./TF4OT/run_all.sh &
```

### Generate plots (evals/plots)
```bash
comparison_box_plots_1d.ipynb
```
```bash
comparison_box_plots_2d.ipynb
```
```bash
python3 TF4OT/evals/plots/generate_plots.py
```
```bash
python3 TF4OT/scripts/generate_feasibility_plots.py ./experiments/cost_semivariable/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data600K_rowwise_costsemivariable_prec4_bins30_DigitTokenizer_epochs25_batch128
```

## 🌟 Citation
See the repository [README](../README.md).