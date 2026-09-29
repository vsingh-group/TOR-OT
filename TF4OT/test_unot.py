import sys
import os
import time
import yaml
import torch
import numpy as np

# Suppress debug logs
import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

# Setup path for UNOT
sys.path.append("/home/ivan/repos/UNOT")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load UNOT
cwd_bkp = os.getcwd()
os.chdir("/home/ivan/repos/UNOT")
try:
    from src.evaluation.import_models import load_fno
    from src.ot.sinkhorn import sink_vec_dist
    from src.ot.cost_matrix import fast_get_cost_matrix
    unot_model = load_fno("unot_fno", device=device, size="standard")

except Exception as e:
    print("Could not load UNOT:", e)
    sys.exit(1)
os.chdir(cwd_bkp)

def sinkhorn_iteration_until_error(set_mu, set_nu, cost_matrix, epsilon, v0, maxiter, dist_target, error_bound):
    K = torch.exp(-cost_matrix / epsilon)
    iter_list = []
    for i in range(set_mu.shape[0]):
        mu = set_mu[i]
        nu = set_nu[i]
        v = v0[i]
        dist_t = dist_target[i]
        for k in range(maxiter):
            u = mu / (K @ v)
            v = nu / (K.T @ u)
            G = torch.diag(u) @ K @ torch.diag(v)
            dist = torch.trace(cost_matrix.T @ G)
            if torch.abs(dist - dist_t) / dist_t < error_bound:
                iter_list.append(k)
                break
    return iter_list

def sinkhorn_computation_predictor(set_mu, set_nu, cost_matrix, epsilon, distance_true, predictor, max_time_s):
    K = torch.exp(-cost_matrix / epsilon)
    time_list = []
    with torch.no_grad():
        V_log = predictor(set_mu, set_nu)
        V = torch.exp(V_log).reshape(set_mu.shape[0], -1)
        start_time = time.time()
        for i in range(set_mu.shape[0]):
            mu = set_mu[i]
            nu = set_nu[i]
            v = V[i]
            dist_t = distance_true[i]
            for k in range(1000):
                u = mu / (K @ v)
                v = nu / (K.T @ u)
                G = torch.diag(u) @ K @ torch.diag(v)
                dist = torch.trace(cost_matrix.T @ G)
                if torch.abs(dist - dist_t) / dist_t < 0.01:
                    break
            time_list.append(time.time() - start_time)
            start_time = time.time()  # reset for next batch item if any
    return time_list, []

def sinkhorn_computation_time(set_mu, set_nu, cost_matrix, epsilon, distance_true, v0, max_time_s):
    K = torch.exp(-cost_matrix / epsilon)
    time_list = []
    start_time = time.time()
    for i in range(set_mu.shape[0]):
        mu = set_mu[i]
        nu = set_nu[i]
        v = v0[i]
        dist_t = distance_true[i]
        for k in range(1000):
            u = mu / (K @ v)
            v = nu / (K.T @ u)
            G = torch.diag(u) @ K @ torch.diag(v)
            dist = torch.trace(cost_matrix.T @ G)
            if torch.abs(dist - dist_t) / dist_t < 0.01:
                break
        time_list.append(time.time() - start_time)
        start_time = time.time()
    return time_list, []

import utils
from run_inference import load_model_by_name
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="TF_100K_cost_fixed")
def main(cfg: DictConfig):
    print("Loading ToR-OT Model...")
    model_base_path = "../experiments/cost_fixed/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_sqeuclidean"
    tor_model, tokenizer = load_model_by_name(model_base_path, device=device, cfg=cfg)
    tor_model = tor_model.to(device)
    tor_model.eval()
    
    # 3. Load Dataset
    print("Loading MNIST...")
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Evaluate over max_samples pairs
    max_samples = 3
    results_unot = []
    
    # For ToR-OT slices
    num_slices = 100
    n_bins = 30
    
    for i in range(max_samples):
        print(f"\n--- Sample Pair {i+1} ---")
        idx1, idx2 = np.random.choice(len(mnist), 2, replace=False)
        img1, _ = mnist[idx1]
        img2, _ = mnist[idx2]
        
        img1 = img1.squeeze().numpy()
        img2 = img2.squeeze().numpy()
        
        H, W = img1.shape
        num_pixels = H * W
        
        a_flat = img1.flatten() + 1e-8
        a_flat = a_flat / a_flat.sum()
        b_flat = img2.flatten() + 1e-8
        b_flat = b_flat / b_flat.sum()
        
        a_torch = torch.tensor(a_flat, dtype=torch.float32, device=device)
        b_torch = torch.tensor(b_flat, dtype=torch.float32, device=device)
        
        x_coords = np.linspace(-1, 1, W)
        y_coords = np.linspace(-1, 1, H)
        xx, yy = np.meshgrid(x_coords, y_coords)
        grid = np.stack([xx.flatten(), yy.flatten()], axis=1)
        grid_torch = torch.tensor(grid, dtype=torch.float32, device=device)
        
        c_matrix = fast_get_cost_matrix(W, device=device)
        
        # ------------------
        # Evaluator metrics
        # ------------------
        eps = 1e-2
        ones = torch.ones(1, num_pixels, device=device)
        a_batch = a_torch.unsqueeze(0)
        b_batch = b_torch.unsqueeze(0)
        
        # Ground truth via Sinkhorn large steps
        _, _, dist_true = sink_vec_dist(a_batch, b_batch, c_matrix, eps, ones, 1000)
        
        # UNOT prediction
        print("Running UNOT / Sinkhorn benchmarking...")
        with torch.no_grad():
            v_0_log = unot_model(a_batch, b_batch)
            v_0 = torch.exp(v_0_log).reshape(1, num_pixels)
            
            # 1. Mean iterations to achieve 0.01 rel error
            iters_unot = sinkhorn_iteration_until_error(a_batch, b_batch, c_matrix, eps, v_0, 1000, dist_true, 0.01)
            iters_vanilla = sinkhorn_iteration_until_error(a_batch, b_batch, c_matrix, eps, ones, 1000, dist_true, 0.01)
            
            iu = iters_unot[0] if len(iters_unot) > 0 else 1000
            iv = iters_vanilla[0] if len(iters_vanilla) > 0 else 1000
            print(f"Iterations (0.01 rel error): UNOT={iu}, Vanilla={iv}")
            
            # 2. Sinkhorn Time up to ground truth
            time_pred, err_pred = sinkhorn_computation_predictor(a_batch, b_batch, c_matrix, eps, dist_true, lambda x,y: v_0_log, max_time_s=1.0)
            time_van, err_van = sinkhorn_computation_time(a_batch, b_batch, c_matrix, eps, dist_true, ones, max_time_s=1.0)
            
            tp = time_pred[-1] if len(time_pred) > 0 else float('nan')
            tv = time_van[-1] if len(time_van) > 0 else float('nan')
            print(f"Time (sinkhorn sequence max): UNOT={tp:.4f}s, Vanilla={tv:.4f}s")
            if tp > 0: print(f"Relative Speedup (Vanilla / UNOT Sinkhorn): {tv/tp:.2f}x")
        
        # ToR-OT 1D Slicing
        print(f"Running Sliced ToR-OT ({num_slices} slices)...")
        thetas = np.random.uniform(0, np.pi, num_slices)
        t0_tor = time.time()
        
        with torch.no_grad():
            avg_plan = torch.zeros((num_pixels, num_pixels), device=device)
            for theta in thetas:
                u = torch.tensor([np.cos(theta), np.sin(theta)], dtype=torch.float32, device=device)
                proj = torch.matmul(grid_torch, u)
                
                active_mask = (a_torch > 0) | (b_torch > 0)
                if active_mask.any():
                    active_proj = proj[active_mask]
                    slice_min, slice_max = active_proj.min().item() - 1e-5, active_proj.max().item() + 1e-5
                else:
                    slice_min, slice_max = -1.0, 1.0
                    
                bins = torch.linspace(slice_min, slice_max, n_bins + 1, device=device)
                bin_indices = torch.bucketize(proj, bins[1:-1])
                
                a_1d = torch.zeros(n_bins, device=device)
                b_1d = torch.zeros(n_bins, device=device)
                a_1d.scatter_add_(0, bin_indices, a_torch)
                b_1d.scatter_add_(0, bin_indices, b_torch)
                
                input_total = torch.cat([a_1d, b_1d])
                if len(input_total) < cfg.data.dataset.input_total_dim:
                    padding = torch.zeros(cfg.data.dataset.input_total_dim - len(input_total), device=device)
                    input_total = torch.cat([input_total, padding])
                input_total = input_total.unsqueeze(0)
                
                x_norm = torch.linspace(-1, 1, n_bins, device=device).view(-1, 1)
                C_1d = torch.cdist(x_norm, x_norm, p=2)**2
                encoder_input_c = C_1d.flatten().unsqueeze(0)
                
                text_ids_pred = tor_model.generate(
                    encoder_input_ab=input_total,
                    encoder_input_c=encoder_input_c,
                    temperature=0.1,
                    max_seq_len=cfg.inference.max_seq_len
                )
                text_pred = tokenizer.from_tokens(text_ids_pred)[0]
                if ']' in text_pred: text_pred = text_pred[:text_pred.find(']')+1]
                ot_pred_dense = utils.parse_predicted_sparse_row_string(text_pred, n_bins, mode='index', precision=cfg.tokenizer.DigitTokenizer.decimal_precision)
                
                if ot_pred_dense == '[]':
                    pi_1d_torch = torch.zeros((n_bins, n_bins), device=device)
                else:
                    pi_1d = np.fromstring(ot_pred_dense[1:-1], sep=',')
                    valid_len = min(len(pi_1d), n_bins * n_bins)
                    padded = np.zeros(n_bins * n_bins)
                    padded[:valid_len] = pi_1d[:valid_len]
                    pi_1d_torch = torch.tensor(padded.reshape(n_bins, n_bins), dtype=torch.float32, device=device)
                    
                a_ratio = torch.zeros_like(a_torch)
                valid_a = a_1d[bin_indices] > 0
                a_ratio[valid_a] = a_torch[valid_a] / a_1d[bin_indices][valid_a]
                
                b_ratio = torch.zeros_like(b_torch)
                valid_b = b_1d[bin_indices] > 0
                b_ratio[valid_b] = b_torch[valid_b] / b_1d[bin_indices][valid_b]
                
                row_bins = bin_indices.unsqueeze(1).expand(-1, num_pixels)
                col_bins = bin_indices.unsqueeze(0).expand(num_pixels, -1)
                
                pi_lifted = pi_1d_torch[row_bins, col_bins] * a_ratio.unsqueeze(1) * b_ratio.unsqueeze(0)
                avg_plan += pi_lifted
                
            avg_plan /= num_slices
        t1_tor = time.time()
        
        tor_dist = (avg_plan * c_matrix).sum().item()
        tor_rel_err = abs(tor_dist - dist_true.item()) / max(1e-9, dist_true.item())
        
        print(f"ToR-OT Execution Time: {t1_tor - t0_tor:.4f}s")
        print(f"ToR-OT Relative Error (vs true Sinkhorn): {tor_rel_err:.4f}")
    
        results_unot.append({
            'iters_unot': iu,
            'iters_van': iv,
            'time_unot': tp,
            'time_van': tv,
            'tor_time': t1_tor - t0_tor,
            'tor_rel_err': tor_rel_err
        })
    
    print("\n" + "="*50)
    print("FINAL AVERAGE RESULTS OVER", max_samples, "PAIRS")
    print("="*50)
    avg_iters_unot = np.mean([r['iters_unot'] for r in results_unot])
    avg_iters_van = np.mean([r['iters_van'] for r in results_unot])
    print(f"Mean Iterations (0.01 error): \n  UNOT={avg_iters_unot:.1f}\n  Vanilla Sinkhorn={avg_iters_van:.1f}")
    
    avg_time_unot = np.mean([r['time_unot'] for r in results_unot])
    avg_time_van = np.mean([r['time_van'] for r in results_unot])
    print(f"Relative Speedup of Sinkhorn with UNOT: {avg_time_van / max(1e-9, avg_time_unot):.2f}x")
    
    avg_tor_time = np.mean([r['tor_time'] for r in results_unot])
    avg_tor_err = np.mean([r['tor_rel_err'] for r in results_unot])
    print(f"ToR-OT Sliced OT Inference Time:   {avg_tor_time:.4f}s")
    print(f"ToR-OT Sliced OT Relative Error:   {avg_tor_err:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
