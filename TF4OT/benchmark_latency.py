import torch
import time
import os
import hydra
from omegaconf import DictConfig
import numpy as np
import ot
from TF4OT.run_inference import instantiate_model, get_tokenizer, get_loaders
from TF4OT import utils
from TF4OT.run_inference_sinkhorn import apply_sinkhorn_projection

# We'll just run a few batches to measure exact speeds natively
try:
    hydra.initialize(config_path="conf", version_base="1.3")
except:
    pass
cfg = hydra.compose(config_name="TF_100K_cost_fixed.yaml")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Model
tokenizer = get_tokenizer(cfg)
n_bins = cfg.dataset.dataset_kwargs.n_bins
precision = cfg.tokenizer.DigitTokenizer.decimal_precision
pytorch_model = instantiate_model(cfg, tokenizer).to(device)

model_dir = "experiments/cost_fixed/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_sqeuclidean"
model_path = os.path.join(model_dir, "best_model.pth")
pytorch_model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
pytorch_model.eval()

# Dummy input
batch_size = 20
L = 30
encoder_input_ab = torch.randint(0, 10, (batch_size, 20)).to(device)
encoder_input_c = torch.randint(0, 10, (batch_size, 5)).to(device)

print(f"\nRunning 5 benchmark iterations on batch_size={batch_size}...")
print(f"{'Iteration':<10} | {'Autoregressive Generate Time':<30} | {'Sinkhorn Math Time':<20} | {'Total Time':<20}")
print("-" * 80)

# Warmup
with torch.no_grad():
    pytorch_model.generate(encoder_input_ab, encoder_input_c, max_seq_len=10)

times_gen = []
times_sink = []

for i in range(5):
    # Base Generation
    start_gen = time.time()
    with torch.no_grad():
        text_ids_pred = pytorch_model.generate(
            encoder_input_ab=encoder_input_ab,
            encoder_input_c=encoder_input_c,
            temperature=0.1,
            max_seq_len=200
        )
    gen_time = time.time() - start_gen
    times_gen.append(gen_time)
    
    # Dummy sinkhorn input
    ot_pred_dense = np.random.rand(batch_size, 30, 30)
    source = np.ones(30)/30.0
    target = np.ones(30)/30.0
    
    start_sink = time.time()
    proj = apply_sinkhorn_projection(ot_pred_dense, source, target)
    sink_time = time.time() - start_sink
    times_sink.append(sink_time)
    
    print(f"{i+1:<10} | {gen_time:<30.4f} | {sink_time:<20.6f} | {gen_time+sink_time:<20.4f}")

print("-" * 80)
print(f"AVERAGE    | {np.mean(times_gen):<30.4f} | {np.mean(times_sink):<20.6f} | {np.mean(times_gen) + np.mean(times_sink):<20.4f}")
