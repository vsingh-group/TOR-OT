# TF4OT: Transformers for Optimal Transport.
### Code for the paper: _"Transformers learn to approximately solve Optimal Transport"_.

## ⚙️ Python Environment Setup

### Setup via conda environment
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda create --name env_TF4OT python=3.11
conda activate env_TF4OT
pip3 install -e .
pip3 install -r requirements.txt
```

### Setup via docker
```bash
docker build -f image.dockerfile -t ivanjaenm/tf4ot:v1 . --push
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

## 🚀 Training Pipeline Overview

### Run training (gpu)
```bash
CUDA_VISIBLE_DEVICES=7 python3 -m TF4OT.run_training --config-name TF_100K_cost_fixed
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m TF4OT.run_training --config-name TF_100K_cost_fixed
```

## 📊 Evaluation

### Run inference (gpu)
```bash
CUDA_VISIBLE_DEVICES=7 python3 -m TF4OT.run_inference --config-name TF_100K_cost_fixed
```

### Run tests
```bash
python3 -m TF4OT.tokenizers_test
python3 -m TF4OT.vocabs_test
```

### Run all experiments in the background
```bash
./TF4OT/run_all.sh &
```

## 🌟 Citation
If you find this work is helpful to your research, please consider citing our paper:
```
@inproceedings{,
  title={Transformers learn to approximately solve Optimal Transport},
  author={},
  booktitle={ArXiv},
  year={2025}
}
```