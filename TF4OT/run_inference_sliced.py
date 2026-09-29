import sys
import os
import torch
import numpy as np
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from typing import NoReturn
import time
from scipy import sparse

# Local imports
from TF4OT import utils
from TF4OT.run_inference import load_model_by_name
import ot

logger = logging.getLogger(__name__)

def eval_sliced_ot_mnist(cfg, pytorch_model, tokenizer, device, num_slices=100, n_bins=30):
    try:
        from torchvision import datasets, transforms
    except ImportError:
        logger.error("torchvision is required. Install it using: pip install torchvision")
        sys.exit(1)

    logger.info("Loading MNIST dataset for sliced OT inference...")
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Take two random images as source and target
    idx1, idx2 = np.random.choice(len(mnist), 2, replace=False)
    img1, _ = mnist[idx1]
    img2, _ = mnist[idx2]

    img1 = img1.squeeze().numpy()
    img2 = img2.squeeze().numpy()

    H, W = img1.shape
    num_pixels = H * W

    # Add a small epsilon and normalize to valid probability distributions
    a = img1.flatten() + 1e-8
    a = a / a.sum()
    b = img2.flatten() + 1e-8
    b = b / b.sum()

    # Create grid points normalized to [-1, 1]
    x_coords = np.linspace(-1, 1, W)
    y_coords = np.linspace(-1, 1, H)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid = np.stack([xx.flatten(), yy.flatten()], axis=1)  # (H*W, 2)
    grid_torch = torch.tensor(grid, dtype=torch.float32, device=device)

    a_torch = torch.tensor(a, dtype=torch.float32, device=device)
    b_torch = torch.tensor(b, dtype=torch.float32, device=device)

    # Accumulator for the lifted OT plan
    avg_plan = torch.zeros((num_pixels, num_pixels), device=device)

    # Generate random slice angles
    thetas = np.random.uniform(0, np.pi, num_slices)

    pytorch_model.eval()
    
    logger.info(f"Running inference over {num_slices} slices...")
    
    # Pre-compute exact POT to evaluate the relative error
    logger.info("Computing exact OT with POT for tracking relative error...")
    C_exact = torch.cdist(grid_torch, grid_torch, p=2)**2
    C_exact_np = C_exact.cpu().numpy()
    
    pi_exact = ot.emd(a, b, C_exact_np)
    exact_ot_cost = np.sum(pi_exact * C_exact_np)
    logger.info(f"Exact POT OT distance: {exact_ot_cost:.6f}")
    
    t0 = time.time()
    time_to_0_01_err = None
    slices_to_0_01_err = None
    
    for i, theta in enumerate(thetas):
        u = torch.tensor([np.cos(theta), np.sin(theta)], dtype=torch.float32, device=device)
        
        # Project grid points onto the slice line
        proj = torch.matmul(grid_torch, u)  # Shape (H*W,)
        
        # Compute dynamic slice bounds based on active mass to maximize model resolution
        active_mask = (a_torch > 0) | (b_torch > 0)
        if active_mask.any():
            active_proj = proj[active_mask]
            slice_min = active_proj.min().item() - 1e-5
            slice_max = active_proj.max().item() + 1e-5
        else:
            slice_min, slice_max = -1.0, 1.0
        bins = torch.linspace(slice_min, slice_max, n_bins + 1, device=device)
        
        # Bin indices (subtract 1 so they are 0-indexed)
        bin_indices = torch.bucketize(proj, bins[1:-1])
        
        # Aggregate mass into bins for source and target
        a_1d = torch.zeros(n_bins, device=device)
        b_1d = torch.zeros(n_bins, device=device)
        
        a_1d.scatter_add_(0, bin_indices, a_torch)
        b_1d.scatter_add_(0, bin_indices, b_torch)
        
        # Run 1D predictor
        input_total = torch.cat([a_1d, b_1d])
        input_total_dim = cfg.data.dataset.input_total_dim
        if len(input_total) < input_total_dim:
            padding = torch.zeros(input_total_dim - len(input_total), device=device)
            input_total = torch.cat([input_total, padding])
        input_total = input_total.unsqueeze(0)
        
        with torch.no_grad():
            encoder_input_ab = input_total
            
            # Use cost metric from config if semivariable, else None or fixed
            cost_type = cfg.data.dataset.input_cost_type
            if cost_type == 'semivariable':
                # Create a one-hot encoding for the configured metric
                target_metric = cfg.data.dataset.input_cost_metric
                metric_idx = utils.metrics.index(target_metric) if target_metric in utils.metrics else 0
                onehot = torch.zeros(1, len(utils.metrics), device=device)
                onehot[0, metric_idx] = 1.0
                encoder_input_c = onehot
            elif cost_type == 'variable':
                # We need a cost matrix for 1D. Let's create a generic squared euclidean cost matrix for [-1, 1].
                x_norm = torch.linspace(-1, 1, n_bins, device=device).view(-1, 1)
                C_1d = torch.cdist(x_norm, x_norm, p=2)**2
                encoder_input_c = C_1d.flatten().unsqueeze(0)
            else:
                encoder_input_c = None
            
            text_ids_pred = pytorch_model.generate(
                encoder_input_ab=encoder_input_ab,
                encoder_input_c=encoder_input_c,
                temperature=0.1,
                max_seq_len=cfg.inference.max_seq_len
            )
            
            text_pred_list = tokenizer.from_tokens(text_ids_pred)
            text_pred = text_pred_list[0]
            if ']' in text_pred:
                text_pred = text_pred[:text_pred.find(']')+1]
                
            ot_pred_dense = utils.parse_predicted_sparse_row_string(
                text_pred, n_bins, mode='index', 
                precision=cfg.tokenizer.DigitTokenizer.decimal_precision
            )
            
            if ot_pred_dense == '[]':
                # Model failed to generate valid sequence
                pi_1d_torch = torch.zeros((n_bins, n_bins), device=device)
            else:
                pi_1d = np.fromstring(ot_pred_dense[1:-1], sep=',')
                if len(pi_1d) != n_bins * n_bins:
                    # Pad or crop if size mismatches
                    valid_len = min(len(pi_1d), n_bins * n_bins)
                    padded = np.zeros(n_bins * n_bins)
                    padded[:valid_len] = pi_1d[:valid_len]
                    pi_1d = padded
                pi_1d = pi_1d.reshape(n_bins, n_bins)
                pi_1d_torch = torch.tensor(pi_1d, dtype=torch.float32, device=device)
        
        # Post-process pi_1d_torch with Sinkhorn-Knopp to precisely enforce marginals
        # The NN output might have slight marginal violations, which blow up during lifting's division by a_1d and b_1d.
        K = pi_1d_torch + 1e-12
        u = torch.ones(n_bins, device=device) / n_bins
        for _ in range(2):
            v = b_1d / (torch.matmul(K.t(), u) + 1e-12)
            u = a_1d / (torch.matmul(K, v) + 1e-12)
        pi_1d_torch = u.unsqueeze(1) * K * v.unsqueeze(0)
        
        # Lift 1D coupling to original 2D grid
        # pi_lifted(i, j) = pi_1d(k_i, l_j) * (a(i) / a_1d(k_i)) * (b(j) / b_1d(l_j))
        
        # Compute ratios safely
        a_ratio = torch.zeros_like(a_torch)
        valid_a = a_1d[bin_indices] > 0
        a_ratio[valid_a] = a_torch[valid_a] / a_1d[bin_indices][valid_a]
        
        b_ratio = torch.zeros_like(b_torch)
        valid_b = b_1d[bin_indices] > 0
        b_ratio[valid_b] = b_torch[valid_b] / b_1d[bin_indices][valid_b]
        
        # We can form pi_lifted efficiently
        # Since pi_1d_torch is 30x30, indexing it with (784, 784) grid is feasible
        row_bins = bin_indices.unsqueeze(1).expand(-1, num_pixels)
        col_bins = bin_indices.unsqueeze(0).expand(num_pixels, -1)
        
        pi_lifted = pi_1d_torch[row_bins, col_bins] * a_ratio.unsqueeze(1) * b_ratio.unsqueeze(0)
        
        avg_plan += pi_lifted
        
        # Check error
        curr_plan_np = (avg_plan / (i + 1)).cpu().numpy()
        est_ot_cost = np.sum(curr_plan_np * C_exact_np)
        rel_err = abs(est_ot_cost - exact_ot_cost) / exact_ot_cost
        logger.info(f"Est OT distance: {est_ot_cost:.6f}")
        logger.info(f"Exact OT distance: {exact_ot_cost:.6f}")
        if rel_err <= 0.01 and time_to_0_01_err is None:
            time_to_0_01_err = time.time() - t0
            slices_to_0_01_err = i + 1
            logger.info(f"Achieved 0.01 relative error at slice {slices_to_0_01_err} in {time_to_0_01_err:.2f} seconds! (Est OT distance: {est_ot_cost:.6f})")
        
        logger.info(f"Processed {i + 1}/{num_slices} slices... Current rel_err: {rel_err:.4f}")

    t1 = time.time()
    sliced_time = t1 - t0
    logger.info(f"Completed {num_slices} slices in {sliced_time:.2f} seconds.")

    if time_to_0_01_err is None:
        logger.info("Did not achieve 0.01 relative error within the given number of slices.")

    avg_plan /= num_slices
    avg_plan_np = avg_plan.cpu().numpy()

    # Visualization
    logger.info("Generating visualizations...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img1, cmap='Blues', aspect='auto')
    axes[0].set_title("Source Image")
    axes[0].axis('off')

    axes[1].imshow(img2, cmap='Blues', aspect='auto')
    axes[1].set_title("Target Image")
    axes[1].axis('off')
    
    axes[2].imshow(avg_plan_np, cmap='viridis', aspect='auto')
    axes[2].set_title("Sliced OT Coupling matrix")
    
    # Compute marginals of the estimated OT plan to see if they match source/target
    pred_marg_a = avg_plan_np.sum(axis=1).reshape(H, W)
    pred_marg_b = avg_plan_np.sum(axis=0).reshape(H, W)
    
    axes[3].imshow(pred_marg_b, cmap='Blues', aspect='auto')
    axes[3].set_title("Target Image (Projected via Plan)")
    axes[3].axis('off')

    plt.tight_layout()
    
    subfolder_name = utils.get_subfolder_name_from_cfg(cfg)
    model_base_path = os.path.join(cfg.model.save_path, subfolder_name)
    save_path = os.path.join(model_base_path, "sliced_inference_mnist.png")
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {save_path}")

def eval_shifted_ot_1d_large(cfg, pytorch_model, tokenizer, device, num_slices=500, n_bins=30):
    # Create two 1D distributions evaluated on a fine grid of 500 points
    num_pixels = 500
    grid = torch.linspace(0, 1, num_pixels, device=device)
    
    # Well-known family 1: Beta(2, 5)
    from torch.distributions import Beta, Normal
    m_a = Beta(torch.tensor([2.0], device=device), torch.tensor([5.0], device=device))
    a_torch = torch.exp(m_a.log_prob(grid))
    a_torch = a_torch / a_torch.sum()
    
    # Well-known family 2: Beta(5, 2)
    m_b = Beta(torch.tensor([5.0], device=device), torch.tensor([2.0], device=device))
    b_torch = torch.exp(m_b.log_prob(grid))
    b_torch = b_torch / b_torch.sum()
    
    avg_plan = torch.zeros((num_pixels, num_pixels), device=device)
    pytorch_model.eval()
    
    # Since it's 1D, angular slices don't apply. Instead, we "slice" by generating randomly shifted bin boundaries.
    # The grid is [0, 1]. The natural bin width is 1.0 / n_bins.
    bin_width = 1.0 / n_bins
    # We randomly shift the start point of the bins between -bin_width and 0
    shifts = np.random.uniform(-bin_width, 0, num_slices)
    
    logger.info(f"Running inference over {num_slices} shifted grids for 1D...")
    
    # Pre-compute exact POT to evaluate the relative error
    logger.info("Computing exact OT with POT for tracking relative error...")
    C_exact = torch.cdist(grid.view(-1, 1), grid.view(-1, 1), p=2)**2
    C_exact_np = C_exact.cpu().numpy()
    a_np = a_torch.cpu().numpy()
    b_np = b_torch.cpu().numpy()
    
    pi_exact = ot.emd(a_np, b_np, C_exact_np)
    exact_ot_cost = np.sum(pi_exact * C_exact_np)
    logger.info(f"Exact POT OT distance: {exact_ot_cost:.6f}")
    
    t0 = time.time()
    time_to_0_01_err = None
    slices_to_0_01_err = None
    
    for i, shift in enumerate(shifts):
        # Create shifted bins that cover [0, 1] fully
        bins = torch.linspace(shift, shift + bin_width * (n_bins + 2), n_bins + 1, device=device)
        
        # Determine which bin each of the 500 grid points falls into
        bin_indices = torch.bucketize(grid, bins[1:-1])
        
        # Clamp to ensure safe indices
        bin_indices = torch.clamp(bin_indices, 0, n_bins - 1)
        
        a_1d = torch.zeros(n_bins, device=device)
        b_1d = torch.zeros(n_bins, device=device)
        
        a_1d.scatter_add_(0, bin_indices, a_torch)
        b_1d.scatter_add_(0, bin_indices, b_torch)
        
        # Run 1D predictor
        input_total = torch.cat([a_1d, b_1d])
        input_total_dim = cfg.data.dataset.input_total_dim
        if len(input_total) < input_total_dim:
            padding = torch.zeros(input_total_dim - len(input_total), device=device)
            input_total = torch.cat([input_total, padding])
        input_total = input_total.unsqueeze(0)
        
        with torch.no_grad():
            encoder_input_ab = input_total
            cost_type = cfg.data.dataset.input_cost_type
            if cost_type == 'semivariable':
                target_metric = cfg.data.dataset.input_cost_metric
                metric_idx = utils.metrics.index(target_metric) if target_metric in utils.metrics else 0
                onehot = torch.zeros(1, len(utils.metrics), device=device)
                onehot[0, metric_idx] = 1.0
                encoder_input_c = onehot
            elif cost_type == 'variable':
                x_norm = torch.linspace(0, 1, n_bins, device=device).view(-1, 1)
                C_1d = torch.cdist(x_norm, x_norm, p=2)**2
                encoder_input_c = C_1d.flatten().unsqueeze(0)
            else:
                encoder_input_c = None
            
            text_ids_pred = pytorch_model.generate(
                encoder_input_ab=encoder_input_ab,
                encoder_input_c=encoder_input_c,
                temperature=0.1,
                max_seq_len=cfg.inference.max_seq_len
            )
            text_pred = tokenizer.from_tokens(text_ids_pred)[0]
            if ']' in text_pred:
                text_pred = text_pred[:text_pred.find(']')+1]
            ot_pred_dense = utils.parse_predicted_sparse_row_string(
                text_pred, n_bins, mode='index', precision=cfg.tokenizer.DigitTokenizer.decimal_precision
            )
            
            if ot_pred_dense == '[]':
                pi_1d_torch = torch.zeros((n_bins, n_bins), device=device)
            else:
                pi_1d = np.fromstring(ot_pred_dense[1:-1], sep=',')
                valid_len = min(len(pi_1d), n_bins * n_bins)
                padded = np.zeros(n_bins * n_bins)
                padded[:valid_len] = pi_1d[:valid_len]
                pi_1d = padded.reshape(n_bins, n_bins)
                pi_1d_torch = torch.tensor(pi_1d, dtype=torch.float32, device=device)
        
        # Sinkhorn-Knopp to enforce marginals
        K = pi_1d_torch + 1e-12
        u = torch.ones(n_bins, device=device) / n_bins
        for _ in range(1):
            v = b_1d / (torch.matmul(K.t(), u) + 1e-12)
            u = a_1d / (torch.matmul(K, v) + 1e-12)
        pi_1d_torch = u.unsqueeze(1) * K * v.unsqueeze(0)
        
        # Lift 1D coupling to 500x500
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
        
        # Check error
        curr_plan_np = (avg_plan / (i + 1)).cpu().numpy()
        est_ot_cost = np.sum(curr_plan_np * C_exact_np)
        logger.info(f"Est OT distance: {est_ot_cost:.6f}")
        logger.info(f"Exact OT distance: {exact_ot_cost:.6f}")
        rel_err = abs(est_ot_cost - exact_ot_cost) / exact_ot_cost
        
        if rel_err <= 0.01 and time_to_0_01_err is None:
            time_to_0_01_err = time.time() - t0
            slices_to_0_01_err = i + 1
            logger.info(f"Achieved 0.01 relative error at shift {slices_to_0_01_err} in {time_to_0_01_err:.2f} seconds! (Est OT distance: {est_ot_cost:.6f}), (Exact OT distance: {exact_ot_cost:.6f}), (Est OT distance: {est_ot_cost:.4f})")
        
        logger.info(f"Processed {i + 1}/{num_slices} shifted grids... Current rel_err: {rel_err:.4f}")

    t1 = time.time()
    sliced_time = t1 - t0
    logger.info(f"Completed {num_slices} shifts in {sliced_time:.2f} seconds.")
    
    if time_to_0_01_err is None:
        logger.info("Did not achieve 0.01 relative error within the given number of slices/shifts.")

    avg_plan /= num_slices
    avg_plan_np = avg_plan.cpu().numpy()
    
    logger.info("Generating visualizations...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(grid.cpu().numpy(), a_torch.cpu().numpy(), label='Source (Beta 2,5)', color='blue')
    axes[0].plot(grid.cpu().numpy(), b_torch.cpu().numpy(), label='Target (Beta 5,2)', color='red')
    axes[0].set_title("Source and Target 1D Distributions (N=500)")
    axes[0].legend()
    
    axes[1].imshow(avg_plan_np, cmap='viridis', aspect='auto', origin='lower')
    axes[1].set_title(f"Shifted OT Coupling matrix ({num_pixels}x{num_pixels})")
    
    pred_marg_b = avg_plan_np.sum(axis=0)
    axes[2].plot(grid.cpu().numpy(), b_torch.cpu().numpy(), label='True Target', color='red', linestyle='--')
    axes[2].plot(grid.cpu().numpy(), pred_marg_b, label='Predicted Target Marginal', color='green')
    axes[2].set_title("Target Marginal Reconstruction")
    axes[2].legend()

    plt.tight_layout()
    
    subfolder_name = utils.get_subfolder_name_from_cfg(cfg)
    model_base_path = os.path.join(cfg.model.save_path, subfolder_name)
    save_path = os.path.join(model_base_path, "shifted_inference_1d_large.png")
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {save_path}")

def inference_routine(cfg):
    """
    Do inference on the saved model for Sliced/Shifted OT.
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        force=True)

    device = torch.device(
        getattr(cfg.inference, "device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.backends.cudnn.benchmark = device.type == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    name = cfg.model.get("name", "model")
    model_name = name if name else "model"

    os.makedirs(cfg.model.save_path, exist_ok=True)

    subfolder_name  = utils.get_subfolder_name_from_cfg(cfg)
    # The actual models are located in the main experiments folder
    model_base_path = os.path.join("/home/ivan/repos/TF4OT/experiments/cost_fixed", subfolder_name)
    log_full_path   = os.path.join(cfg.model.save_path, subfolder_name, "log_inference_sliced.log")

    utils.setup_logging(log_full_path, 0)

    logger.info(f"Loading model from: {model_base_path}")
    pytorch_model, tokenizer = load_model_by_name(model_base_path, cfg=cfg)

    n_bins = cfg.data.dataset.support_size
    num_slices = getattr(cfg.inference, "num_slices", 500)

    logger.info(f"{'='*80}")
    logger.info("Starting Sliced OT Evaluation on MNIST Distributions...")
    logger.info(f"{'='*80}\n")
    
    eval_sliced_ot_mnist(cfg, pytorch_model=pytorch_model, tokenizer=tokenizer, device=device, num_slices=num_slices, n_bins=n_bins)
    # eval_shifted_ot_1d_large(cfg, pytorch_model=pytorch_model, tokenizer=tokenizer, device=device, num_slices=num_slices, n_bins=n_bins)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logger.info("Using GPU inference mode")
    if cfg.training.world_size is None: 
        cfg.training.world_size = torch.cuda.device_count()

    inference_routine(cfg)

if __name__ == "__main__":
    main()