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

### Run training
```console
CUDA_VISIBLE_DEVICES=7 python3 -m TF4OT.run_training --config-name small
```

### Run inference
```
CUDA_VISIBLE_DEVICES=7 python3 -m TF4OT.run_inference --config-name small
```

### Run tests
```console
python3 -m TF4OT.tokenizers_test
python3 -m TF4OT.vocabs_test
```