import torch
import time
import os
import hydra
from omegaconf import DictConfig
import numpy as np
from TF4OT.run_inference import instantiate_model, get_tokenizer, get_loaders
from TF4OT.run_inference_sinkhorn import apply_sinkhorn_projection

try:
    hydra.initialize(config_path="conf", version_base="1.3")
except:
    pass
cfg = hydra.compose(config_name="TF_100K_cost_fixed.yaml")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = get_tokenizer(cfg)
n_bins = cfg.dataset.dataset_kwargs.n_bins
precision = cfg.tokenizer.DigitTokenizer.decimal_precision
pytorch_model = instantiate_model(cfg, tokenizer).to(device)

model_dir = "experiments/cost_fixed/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_sqeuclidean"
model_path = os.path.join(model_dir, "best_model.pth")
pytorch_model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
pytorch_model.eval()

_, test_loader = get_loaders(cfg, args=None)

times_gen = []
times_sink = []

print("Running 10 batches of testset equivalent inference...", flush=True)

for i, batch in enumerate(test_loader):
    if i >= 10: break
    
    encoder_input_ab = batch['input_total_vec'].to(device)
    encoder_input_c = None
    
    start_gen = time.time()
    with torch.no_grad():
        text_ids_pred = pytorch_model.generate(
            encoder_input_ab=encoder_input_ab,
            encoder_input_c=encoder_input_c,
            temperature=0.1,
            max_seq_len=cfg.inference.max_seq_len
        )
    gen_time = time.time() - start_gen
    
    # Per instance time inside the batch
    b_size = encoder_input_ab.shape[0]
    time_per_instance_gen = gen_time / b_size
    times_gen.append(time_per_instance_gen)
    
    # Dummy sinkhorn to compute the time
    ot_pred_dense = np.random.rand(b_size, 30, 30)
    source = np.ones(30)/30.0
    target = np.ones(30)/30.0
    
    start_sink = time.time()
    proj = apply_sinkhorn_projection(ot_pred_dense, source, target)
    sink_time = time.time() - start_sink
    time_per_instance_sink = sink_time / b_size
    times_sink.append(time_per_instance_sink)
    
    print(f"Batch {i}: Gen {time_per_instance_gen:.5f}s/item | Sink {time_per_instance_sink:.5f}s/item")

avg_gen = np.mean(times_gen)
avg_sink = np.mean(times_sink)

np.save(os.path.join(model_dir, "testset_speeds.npy"), {"can": avg_gen, "sink": avg_gen + avg_sink})
print("Saved speeds successfully!", flush=True)
