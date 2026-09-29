import os
import torch
import numpy as np
from datasets import load_dataset
from scipy import sparse
from tqdm import tqdm
from tabulate import tabulate

# Add TF4OT root to paths if necessary
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_inference import load_model_by_name
import utils
from omegaconf import OmegaConf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import yaml

# Path to the specific experiment model dir
model_base_path = "../experiments/cost_fixed/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_sqeuclidean"

print(f"Loading config from configs/TF_100K_cost_fixed.yaml...")
yaml_config_path = "configs/TF_100K_cost_fixed.yaml"

with open(yaml_config_path) as f:
    cfg = yaml.safe_load(f)

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            if isinstance(my_dict[key], dict):
                setattr(self, key, Dict2Class(my_dict[key]))
            else:
                setattr(self, key, my_dict[key])
cfg = Dict2Class(cfg)

n_bins = cfg.data.dataset.support_size
max_samples = 50

# Load model and tokenizer
print("Loading model and tokenizer...")
pytorch_model, tokenizer = load_model_by_name(model_base_path)
pytorch_model = pytorch_model.to(device)
pytorch_model.eval()

# Load dataset
print(f"Loading dataset: {cfg.data.hf_dataset} ...")
ds = load_dataset(cfg.data.hf_dataset, split='test')
ds = ds.select(range(max_samples))

print("Running inference...")
results = []
cost_type = cfg.data.dataset.input_cost_type

# Variables for mapping keys correctly
OT_INPUT_TOTAL = "input_total"
OT_INPUT_COST_MATRIX = "input_cost_matrix"
OT_INPUT_COST_ONEHOT = "input_cost_type"

for idx in tqdm(range(len(ds))):
    sample = ds[idx]
    
    input_total_vec = sample[OT_INPUT_TOTAL]
    source_hist = input_total_vec[:n_bins]
    target_hist = input_total_vec[n_bins:2*n_bins]
    
    source_marg = np.array(source_hist, dtype=np.float32)
    target_marg = np.array(target_hist, dtype=np.float32)
    
    # Needs to be shape (1, seq_len)
    encoder_input_ab = torch.tensor(input_total_vec).unsqueeze(0).to(device)
    
    if cost_type == 'fixed':
        encoder_input_c = None
    elif cost_type == 'variable':
        encoder_input_c = torch.tensor(sample[OT_INPUT_COST_MATRIX]).flatten().unsqueeze(0).to(device)
    elif cost_type == 'semivariable':
        encoder_input_c = torch.tensor(sample[OT_INPUT_COST_ONEHOT]).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # ------------- Standard Inference -------------
        text_ids_pred_1 = pytorch_model.generate(
            encoder_input_ab = encoder_input_ab,
            encoder_input_c  = encoder_input_c,
            temperature=0.1,
            max_seq_len=cfg.inference.max_seq_len
        )
        text_pred_1 = tokenizer.from_tokens(text_ids_pred_1)
        text_pred_1 = [t[:t.find(']')+1] if (']' in t) else t for t in text_pred_1]
        text_pred_dense_1 = [utils.parse_predicted_sparse_row_string(t, n_bins, mode='index', precision=cfg.tokenizer.DigitTokenizer.decimal_precision) for t in text_pred_1 ]
        ot_pred_dense_1 = np.stack([np.fromstring(t[1:-1], sep=',') for t in text_pred_dense_1])
        
        # Sparse plan standard
        ot_pred_sparse_std = sparse.coo_matrix(ot_pred_dense_1)
        metrics_std = utils.compute_constraint_violations(ot_pred_sparse_std, source_marg, target_marg)

        # ------------- Symmetrized Inference -------------
        # --- Pass 2: Swapped Direction ---
        source_part = encoder_input_ab[:, :n_bins]
        target_part = encoder_input_ab[:, n_bins:2*n_bins]
        rem_part = encoder_input_ab[:, 2*n_bins:]
        encoder_input_ab_swapped = torch.cat([target_part, source_part, rem_part], dim=1)
        
        if cost_type == 'variable' and encoder_input_c is not None:
            encoder_input_c_swapped = encoder_input_c.view(-1, n_bins, n_bins).transpose(1, 2).flatten(start_dim=1)
        else:
            encoder_input_c_swapped = encoder_input_c

        text_ids_pred_2 = pytorch_model.generate(
            encoder_input_ab = encoder_input_ab_swapped,
            encoder_input_c  = encoder_input_c_swapped,
            temperature=0.1,
            max_seq_len=cfg.inference.max_seq_len
        )
        text_pred_2 = tokenizer.from_tokens(text_ids_pred_2)
        text_pred_2 = [t[:t.find(']')+1] if (']' in t) else t for t in text_pred_2]
        text_pred_dense_2 = [utils.parse_predicted_sparse_row_string(t, n_bins, mode='index', precision=cfg.tokenizer.DigitTokenizer.decimal_precision) for t in text_pred_2 ]
        ot_pred_dense_2 = np.stack([np.fromstring(t[1:-1], sep=',') for t in text_pred_dense_2])
        
        # Symmetrized plan (Average of 1 and transposed 2)
        ot_pred_dense_sym = (ot_pred_dense_1 + ot_pred_dense_2.T) / 2.0
        ot_pred_sparse_sym = sparse.coo_matrix(ot_pred_dense_sym)
        metrics_sym = utils.compute_constraint_violations(ot_pred_sparse_sym, source_marg, target_marg)

        results.append({
            'std_row': metrics_std['row_violation'],
            'std_col': metrics_std['col_violation'],
            'std_max': metrics_std['max_violation'],
            'sym_row': metrics_sym['row_violation'],
            'sym_col': metrics_sym['col_violation'],
            'sym_max': metrics_sym['max_violation']
        })

# Print summary
std_row_mean = np.mean([r['std_row'] for r in results])
std_col_mean = np.mean([r['std_col'] for r in results])
std_max_mean = np.mean([r['std_max'] for r in results])

sym_row_mean = np.mean([r['sym_row'] for r in results])
sym_col_mean = np.mean([r['sym_col'] for r in results])
sym_max_mean = np.mean([r['sym_max'] for r in results])

table = [
    ["Standard (run_inference)", f"{std_row_mean:.5f}", f"{std_col_mean:.5f}", f"{std_max_mean:.5f}"],
    ["Symmetrized (run_inference_sym)", f"{sym_row_mean:.5f}", f"{sym_col_mean:.5f}", f"{sym_max_mean:.5f}"]
]

res_str = "\n=== Mean Constraint Violations (L1 Error) ===\n"
res_str += tabulate(table, headers=["Method", "Mean Row Violation", "Mean Col Violation", "Mean Max Violation"], tablefmt="grid")
print(res_str)

# Write to file so we can view it
with open("constraint_violations_summary.txt", "w") as f:
    f.write(res_str)
