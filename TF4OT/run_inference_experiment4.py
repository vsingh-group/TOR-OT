import os
import time
import logging
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from scipy import sparse
import hydra
from omegaconf import DictConfig

import ot
from tabulate import tabulate

from TF4OT import utils
from TF4OT.run_inference_sym import load_model_by_name, get_cost_matrix

logger = logging.getLogger(__name__)

OT_INPUT_TOTAL         = "input_total"
OT_INPUT_COST_ONEHOT   = "input_cost_type"
OT_INPUT_COST_MATRIX   = "input_cost_matrix"
OT_TARGET_PLAN_MATRIX_SPARSE  = "target_otplan_solve_sparse"

def inference_experiment4(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        force=True)

    device = torch.device(getattr(cfg.inference, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    # Load model
    subfolder_name = utils.get_subfolder_name_from_cfg(cfg)
    model_base_path = os.path.join(cfg.model.save_path, subfolder_name)
    logger.info(f"Loading model from: {model_base_path}")
    pytorch_model, tokenizer = load_model_by_name(model_base_path, device.type)
    pytorch_model.eval()

    # Load dataset
    logger.info(f"Loading dataset: {cfg.data.hf_dataset}")
    dataset = load_dataset(cfg.data.hf_dataset, split="test", cache_dir=cfg.data.hf_cache_dir)
    n_bins = cfg.data.dataset.support_size
    
    # Sample a manageable subset for precise timing
    num_samples = 100
    test_data = dataset.select(range(min(num_samples * n_bins, len(dataset))))
    
    test_loader = DataLoader(test_data,
                             batch_size=n_bins, # Process one full graph at a time
                             shuffle=False,
                             collate_fn=utils.collate_from_string)

    results = []

    # Prepare cost matrix (Assuming sqeuclidean or similar default logic)
    metric = getattr(cfg.data.dataset, "input_cost_metric", "sqeuclidean")
    logger.info(f"Using cost metric: {metric}")
    C_tensor = get_cost_matrix(metric, n_bins, device)
    C_np = C_tensor.cpu().numpy()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_samples: break
            
            # Extract inputs
            input_total_vec = batch[OT_INPUT_TOTAL][0]
            source = input_total_vec[:n_bins].numpy()
            target = input_total_vec[n_bins:2*n_bins].numpy()
            
            # --- METHOD 1: EXACT SOLVER (POT) ---
            t0 = time.perf_counter()
            ot_exact = ot.emd(source, target, C_np)
            time_exact = (time.perf_counter() - t0) * 1000 # ms
            
            exact_cost = (ot_exact * C_np).sum()
            
            # --- METHOD 2: NEURAL MODEL (TF4OT) ---
            encoder_input_ab = batch[OT_INPUT_TOTAL].to(device)
            # handle cost
            cost_type = cfg.data.dataset.input_cost_type
            if cost_type == 'fixed':
                encoder_input_c = None
            elif cost_type == 'semivariable':
                encoder_input_c = batch[OT_INPUT_COST_ONEHOT].flatten(start_dim=1).to(device)
            else:
                encoder_input_c = batch[OT_INPUT_COST_MATRIX].flatten(start_dim=1).to(device)
            
            t0 = time.perf_counter()
            text_ids_pred = pytorch_model.generate(
                encoder_input_ab=encoder_input_ab,
                encoder_input_c=encoder_input_c,
                temperature=0.1,
                max_seq_len=cfg.inference.max_seq_len
            )
            text_pred_list = tokenizer.from_tokens(text_ids_pred)
            text_pred = text_pred_list[0][:text_pred_list[0].find(']')+1] if (']' in text_pred_list[0]) else text_pred_list[0]
            parsed_dense = utils.parse_predicted_sparse_row_string(text_pred, n_bins, mode='index', precision=cfg.tokenizer.DigitTokenizer.decimal_precision)
            # Reconstruct sparse array
            dense_arr = np.fromstring(parsed_dense[1:-1], sep=',')
            ot_pred_sparse = sparse.coo_matrix(dense_arr.reshape((n_bins, n_bins)))
            time_neural = (time.perf_counter() - t0) * 1000 # ms
            
            model_cost = (ot_pred_sparse.toarray() * C_np).sum()
            model_mae = np.abs(ot_exact - ot_pred_sparse.toarray()).mean()

            # --- METHOD 3: SCREENED WARMSTART SOLVER ---
            # Create restricted cost matrix where model predicts 0
            # To ensure strict screening without infinite float issues in C++ simplex,
            # we assign a huge penalty (e.g., 1000 * max(C)) to edges that the model drops.
            C_restricted = C_np.copy()
            pred_dense = ot_pred_sparse.toarray()
            mask = (pred_dense <= 1e-6)
            penalty_val = C_np.max() * 1000.0 if C_np.max() > 0 else 1000.0
            C_restricted[mask] = penalty_val
            
            t0 = time.perf_counter()
            ot_warm = ot.emd(source, target, C_restricted)
            time_warm = (time.perf_counter() - t0) * 1000 # ms
            
            warm_cost = (ot_warm * C_np).sum()
            
            # Check if warmstart found the true exact solution or a penalized one
            success = 1 if abs(warm_cost - exact_cost) < 1e-4 else 0
            
            results.append({
                "exact_time": time_exact,
                "neural_time": time_neural,
                "warm_time": time_warm,
                "model_cost": model_cost,
                "exact_cost": exact_cost,
                "model_mae": model_mae,
                "warm_success": success
            })
            
            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx}/{num_samples} samples...")

    # Aggregate & Report
    avg_exact_t = np.mean([r["exact_time"] for r in results])
    avg_neural_t = np.mean([r["neural_time"] for r in results])
    avg_warm_t = np.mean([r["warm_time"] for r in results])
    
    avg_cost_diff = np.mean([abs(r["model_cost"] - r["exact_cost"]) for r in results])
    avg_mae = np.mean([r["model_mae"] for r in results])
    warm_success_rate = np.mean([r["warm_success"] for r in results]) * 100

    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 4: TIME AND ACCURACY COMPARISON WITH WARMSTART")
    logger.info("="*80)
    
    table = [
        ["Method", "Avg Time (ms)", "OT Cost Diff", "MAE vs Exact", "Warmstart Success"],
        ["Exact OT (POT Simplex)", f"{avg_exact_t:.2f} ms", "-", "-", "-"],
        ["Neural OT (TF4OT)", f"{avg_neural_t:.2f} ms", f"{avg_cost_diff:.3e}", f"{avg_mae:.3e}", "-"],
        ["Neural Warmstart + EXACT", f"{avg_warm_t:.2f} ms", "-", "-", f"{warm_success_rate:.1f}%"]
    ]
    
    logger.info("\n" + tabulate(table, headers="firstrow", tablefmt="grid"))
    logger.info("="*80)

@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):
    inference_experiment4(cfg)

if __name__ == "__main__":
    main()
