# TF4OT: Transformers for Optimal Transport.
### Code for the paper: _"Transformers learn to approximately solve Optimal Transport"_.

### Setup python environment via conda
```console
source ~/anaconda3/etc/profile.d/conda.sh
conda create --name env_TF4OT python=3.11
conda activate env_TF4OT
pip3 install -e .
pip3 install -r requirements.txt
```

### Run data generator (runs on cpu)
```console
python3 -m TF4OT.run_datagenerator --config-name small
python3 -m TF4OT.run_datagenerator --n_bins=10 --row_wise --metric='sqeuclidean' --cost_fixed=True  --dataset_size=100_000 --output_dir=./ot_dataset_row_test --upload --repo_name='ivanjaenm/ot_dataset_row_test'
python3 -m TF4OT.run_datagenerator --n_bins=10 --metric='sqeuclidean' --cost_fixed=True  --dataset_size=100_000 --output_dir=./ot_dataset_test --upload --repo_name='ivanjaenm/ot_dataset_test'
python3 -m TF4OT.run_datagenerator --n_bins=10 --row_wise --metric='sqeuclidean' --cost_fixed=False --dataset_size=100_000 --output_dir=./ot_dataset_cost_row_test --upload --repo_name='ivanjaenm/ot_dataset_cost_row_test'
python3 -m TF4OT.run_datagenerator --n_bins=10 --metric='sqeuclidean' --cost_fixed=False --dataset_size=100_000 --output_dir=./ot_dataset_cost_test --upload --repo_name='ivanjaenm/ot_dataset_cost_test'
```

### Run training (runs on gpu)
```console
CUDA_VISIBLE_DEVICES=7 python3 -m TF4OT.run_training --config-name small
CUDA_VISIBLE_DEVICES=7 python3 -m TF4OT.run_training --config-name test_row_wise
```

### Run inference
```console
CUDA_VISIBLE_DEVICES=7 python3 -m TF4OT.run_inference --config-name small
CUDA_VISIBLE_DEVICES=7 python3 -m TF4OT.run_inference --config-name test_row_wise
```

### Run tests
```console
python3 -m TF4OT.tokenizers_test
python3 -m TF4OT.vocabs_test
```

### TODOs:
#### Enable DDP
#### Multiple tokenizers
#### Add Neural OT experiments
#### Multiple encoders for inputs
