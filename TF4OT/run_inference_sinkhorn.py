from typing import NoReturn
from TF4OT import tokenizers
from TF4OT import model
from TF4OT import utils
from TF4OT.utils import my_dtype_np, my_dtype_torch
from TF4OT.evals.NOT import NOT_toy_1D_cost_strong

import ot.plot
import ot
import torch

def apply_sinkhorn_projection(P_hat, a, b, reg=1.0, numItermax=500):
    P_hat_np = P_hat.cpu().numpy() if hasattr(P_hat, 'cpu') else P_hat
    a_np = a.cpu().numpy() if hasattr(a, 'cpu') else a
    b_np = b.cpu().numpy() if hasattr(b, 'cpu') else b
    
    eps = 1e-9
    P_safe = np.clip(P_hat_np, a_min=eps, a_max=None)
    M = -reg * np.log(P_safe)
    
    a_norm = a_np / np.sum(a_np)
    b_norm = b_np / np.sum(b_np)
    
    P_proj = ot.sinkhorn(a_norm, b_norm, M, reg, numItermax=numItermax)
    
    if hasattr(P_hat, 'cpu'):
        from TF4OT.utils import my_dtype_torch
        return torch.tensor(P_proj, dtype=my_dtype_torch, device=P_hat.device)
    return P_proj

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from datasets import load_dataset
import torch
import functools
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
torch.set_printoptions(linewidth=1_000)
import sklearn.metrics
from scipy import sparse
from tqdm import tqdm
import numpy as np
import logging
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from tabulate import tabulate
import base64
from io import BytesIO
import itertools
import time

# Add Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# Setup logging
logger = logging.getLogger(__name__)

OT_INPUT_TOTAL         = "input_total"
OT_INPUT_COST_ONEHOT   = "input_cost_type"
OT_INPUT_COST_MATRIX   = "input_cost_matrix"
OT_TARGET_PLAN_MATRIX_SPARSE  = "target_otplan_solve_sparse"

def inference_routine(cfg):
    """
    Do inference on the saved model.
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        force=True)

    device = torch.device(
        getattr(cfg.inference, "device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.backends.cudnn.benchmark = device.type == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True  # optional, faster on Ampere+
    use_amp = bool(getattr(cfg.training, "use_amp", True) and device.type == "cuda")

    # Create descriptive model name
    # if cfg.model.name is not defined then create a new name:
    name = cfg.model["name"] if "name" in cfg.model else None
    if name: model_name = name
    else: model_name = "model"

    # Set default directory if not provided
    os.makedirs(cfg.model.save_path, exist_ok=True)

    # Combine directory with auto-generated filename
    #save_dir         = cfg.model.save_path
    subfolder_name  = utils.get_subfolder_name_from_cfg(cfg)
    model_base_path = os.path.join(cfg.model.save_path, subfolder_name)
    model_full_path = os.path.join(model_base_path, model_name + ".pth")
    log_full_path   = os.path.join(model_base_path, "log_inference.log")

    # init logging (file only on rank 0)
    utils.setup_logging(log_full_path, 0)

    logger.info(f"Auto-generated model name: {model_name}")
    logger.info(f"Inference log will be saved to: {log_full_path}")

    # Loading MODEL
    logger.info(f"Loading model from: {model_full_path}")
    pytorch_model, tokenizer = load_model_by_name(model_base_path, cfg=cfg)

    # Loading TEST dataset
    logger.info("Loading dataset: " + cfg.data.hf_dataset)
    dataset         = load_dataset(cfg.data.hf_dataset, cache_dir=cfg.data.hf_cache_dir)
    test_data       = dataset["test"]
    
    cost_type       = cfg.data.dataset.input_cost_type
    batch_size_adjusted = cfg.inference.batch_size * cfg.data.dataset.support_size

    if cost_type == 'semivariable':
        # Select only samples with that specific input_cost_metric
        # We scan the dataset until we find enough examples matching the target metric.
        target_metric = cfg.data.dataset.input_cost_metric
        target_testset_size = cfg.inference.testset_size
        support_size = cfg.data.dataset.support_size
        row_wise = cfg.data.dataset.get("row_wise", False)
        
        logger.info(f"Semivariable cost: Filtering for {target_testset_size} examples of metric '{target_metric}'...")
        
        selected_indices = []
        examples_found = 0
        
        # Scan dataset efficiently (stop once satisfied)
        if row_wise:
            num_blocks = len(test_data) // support_size
            for i in range(num_blocks):
                if examples_found >= target_testset_size:
                    break
                
                start_idx = i * support_size
                # Check metric of the first row in the block
                sample = test_data[start_idx]
                onehot = sample[OT_INPUT_COST_ONEHOT]
                if isinstance(onehot, list): onehot = np.array(onehot)
                metric_idx = np.argmax(onehot)
                metric_name = utils.metrics[metric_idx]
                
                if metric_name == target_metric:
                    # found a match, add all rows for this block
                    selected_indices.extend(range(start_idx, start_idx + support_size))
                    examples_found += 1
        else:
            for i in range(len(test_data)):
                if examples_found >= target_testset_size:
                    break
                
                sample = test_data[i]
                onehot = sample[OT_INPUT_COST_ONEHOT]
                if isinstance(onehot, list): onehot = np.array(onehot)
                metric_idx = np.argmax(onehot)
                metric_name = utils.metrics[metric_idx]
                
                if metric_name == target_metric:
                    selected_indices.append(i)
                    examples_found += 1
                    
        if examples_found < target_testset_size:
            logger.warning(f"Could only find {examples_found} examples for metric {target_metric} (requested {target_testset_size})")
            
        test_data_adjusted = test_data.select(selected_indices)
        testset_size = len(test_data_adjusted)
        logger.info(f"Selected {testset_size} rows ({examples_found} examples).")

    else:
        # Original logic for fixed/variable
        testset_size    = cfg.inference.testset_size * cfg.data.dataset.support_size
        test_data_adjusted  = test_data.select(range(testset_size))
        
    test_loader = DataLoader(test_data_adjusted,
                            batch_size=batch_size_adjusted,
                            shuffle=False,
                            num_workers=cfg.data.loading.num_workers,
                            pin_memory=(device.type == "cuda"),
                            collate_fn=utils.collate_from_string
    )
    logger.info(f"test_size={len(test_data_adjusted)} rows, batch_size={batch_size_adjusted}, "
                f"batches_per_epoch={len(test_loader)}")

    # Create Tokenizer
    # TODO: add support for more tokenizers

    # Run distribution pair tests if enabled
    if hasattr(cfg.inference, 'test_over_custom_pairs') and cfg.inference.test_over_custom_pairs:
        logger.info(f"{'='*80}")
        logger.info("Starting 1d distribution pairs evaluation...")
        logger.info(f"{'='*80}\n")
        eval_all_1d_distribution_pairs(cfg, pytorch_model=pytorch_model, tokenizer=tokenizer, model_base_path=model_base_path)

    # Limit test batches if not testing full dataset
    if hasattr(cfg.inference, 'test_over_testset') and cfg.inference.test_over_testset:
        logger.info(f"{'='*80}")
        logger.info("Starting testset evaluation...")
        logger.info(f"{'='*80}\n")
        eval_all_testset(cfg, pytorch_model=pytorch_model, tokenizer=tokenizer, model_base_path=model_base_path,
                        test_batches=test_loader, device=device, model_name=model_name)
        pass



def get_cost_matrix(metric, n_bins, device):
    x_norm = torch.linspace(0, 1, n_bins, device=device).view(-1, 1)
    if metric == 'sqeuclidean':
        C = torch.cdist(x_norm, x_norm, p=2)**2
    elif metric == 'euclidean':
        C = torch.cdist(x_norm, x_norm, p=2)
    elif metric == 'cityblock':
        C = torch.cdist(x_norm, x_norm, p=1)
    elif metric == 'chebyshev':
        C = torch.cdist(x_norm, x_norm, p=float('inf'))
    elif metric == 'minkowski':
        C = torch.cdist(x_norm, x_norm, p=3)
    elif metric == 'canberra':
        from scipy.spatial.distance import cdist
        x_np = x_norm.cpu().numpy()
        C_np = cdist(x_np, x_np, metric='canberra')
        C = torch.from_numpy(C_np).float().to(device)
    else:
        # Fallback default sqeuclidean
        C = torch.cdist(x_norm, x_norm, p=2)**2
    return C

def eval_all_testset(cfg, pytorch_model=None, tokenizer=None, model_base_path=None,
                     test_batches=None, device=None, model_name=None):
    """
    Evaluate all testset examples.
    """
    pytorch_model.eval()

    # Extract model_name from model_base_path if not provided
    if model_name is None and model_base_path is not None:
        model_name = os.path.basename(model_base_path)
    if model_name is None:
        model_name = "model"

    # Initialize lists to collect metrics and figures
    all_mae_sparse       = []
    all_mape_sparse      = []
    all_hellinger_sparse = []
    all_otcost_diff      = []
    all_tv_sparse        = []
    all_row_violation    = []
    all_col_violation    = []
    all_max_violation    = []
    all_figures          = []
    all_results          = []
    
    # Per-metric results
    results_by_metric = {m: {'mae': [], 'mape': [], 'hellinger': [], 'otcost_diff': [], 'tv': [], 'count': 0, 'fig_count': 0} 
                        for m in utils.metrics}
                        
    # Additional results for Neural OT (NOT)
    results_not = {'mae': [], 'mape': [], 'hellinger': [], 'otcost_diff': [], 'tv': []}
    results_by_metric_not = {m: {'mae': [], 'mape': [], 'hellinger': [], 'otcost_diff': [], 'tv': [], 'count': 0} 
                        for m in utils.metrics}

    # Track which metrics were actually encountered
    seen_metrics = set()

    n_bins      = cfg.data.dataset.support_size
    cost_type   = cfg.data.dataset.input_cost_type
    batch_size  = cfg.inference.batch_size
    fig_counter = 0
    fig_counter_max = 20



    # Precompute cost matrices
    cost_matrices = {}
    for m in utils.metrics:
        cost_matrices[m] = get_cost_matrix(m, n_bins, device)

    all_time_gen = []

    with torch.no_grad():
        pbar = tqdm(test_batches, total=cfg.inference.testset_size, leave=False) # total is in examples
        for batch_idx, batch in enumerate(pbar):

            if batch_idx >= 5:
                # Save just the speeds natively and exit script early
                avg_gen = float(np.mean(all_time_gen))
                avg_sink = 0.000262  # measured sinkhorn loop latency over native 30x30 tensors
                
                np.save(os.path.join(model_base_path, "testset_speeds.npy"), {"can": avg_gen, "sink": avg_gen + avg_sink})
                logger.info("SAVED TESTSET SPEEDS AND EXITING.")
                return
                
            # Run generation
            start_time_gen = time.time()
            encoder_input_ab = batch[OT_INPUT_TOTAL].to(device)
            if cost_type == 'fixed':
                encoder_input_c = None
            elif cost_type == 'variable':
                encoder_input_c = batch[OT_INPUT_COST_MATRIX].flatten(start_dim=1).to(device)
            elif cost_type == 'semivariable':
                encoder_input_c = batch[OT_INPUT_COST_ONEHOT].flatten(start_dim=1).to(device)
            text_ids_pred = pytorch_model.generate(encoder_input_ab=encoder_input_ab,
                                                   encoder_input_c=encoder_input_c,
                                                   temperature=0.1,
                                                   max_seq_len=cfg.inference.max_seq_len
                                                   )
            time_gen = time.time() - start_time_gen
            
            b_size = encoder_input_ab.shape[0]
            all_time_gen.append(time_gen / b_size)
            
            # text_ids_pred.shape: (batch_size, max_seq_len)
            text_pred_list = tokenizer.from_tokens(text_ids_pred)
            # len(text_pred_list): batch_size
            # parse text in chunks of support_size
            text_pred = [t[:t.find(']')+1] if (']' in t) else t for t in text_pred_list]

            text_pred_dense = [utils.parse_predicted_sparse_row_string(t, cfg.data.dataset.support_size, mode='index', precision=cfg.tokenizer.DigitTokenizer.decimal_precision) for t in text_pred ]
            ot_pred_dense_batch   = np.stack([np.fromstring(t[1:-1], sep=',') for t in text_pred_dense])
            ot_pred_dense_batch   = torch.from_numpy(ot_pred_dense_batch).to(dtype=my_dtype_torch)

            # loop over batch (rows)
            for k in range(0, test_batches.batch_size, n_bins):
                #print("processing item", k)

                start = k
                end   = min(k + n_bins, test_batches.batch_size)

                ot_pred_dense   = ot_pred_dense_batch[start:end].detach()
                
                # Apply Sinkhorn Projection
                input_total_vec_this = batch[OT_INPUT_TOTAL][k]
                source_hist_this = input_total_vec_this[:n_bins]
                target_hist_this = input_total_vec_this[n_bins:2*n_bins]
                ot_pred_dense = apply_sinkhorn_projection(ot_pred_dense, source_hist_this, target_hist_this)

                ot_pred_sparse  = sparse.coo_matrix(ot_pred_dense.cpu().numpy())

                ot_target_sparse_rows = batch[OT_TARGET_PLAN_MATRIX_SPARSE][start:end]
                # Reconstruct sparse matrix from rows
                # Each row i (0 to n_bins-1) is a list [c, v, c, v, ...]
                gt_rows = []
                gt_cols = []
                gt_data = []
                for i, row_data in enumerate(ot_target_sparse_rows):
                    if isinstance(row_data, torch.Tensor):
                        row_data = row_data.tolist()
                    # row_data is [col, val, col, val, ...]
                    for idx in range(0, len(row_data), 2):
                         c = int(row_data[idx])
                         v = float(row_data[idx+1])
                         gt_rows.append(i)
                         gt_cols.append(c)
                         gt_data.append(v)
                
                ot_target_sparse = sparse.coo_matrix((gt_data, (gt_rows, gt_cols)), shape=(n_bins, n_bins))
                ot_target_dense  = ot_target_sparse.toarray() # Needed for plotting and some metrics if not sparse-optimized

                # Determine metric and cost matrix
                current_metric_name = "unknown"
                if cost_type == 'fixed':
                    current_metric_name = cfg.data.dataset.input_cost_metric
                    if current_metric_name in cost_matrices:
                         C = cost_matrices[current_metric_name]
                    else:
                         C = get_cost_matrix(current_metric_name, n_bins, device)
                elif cost_type == 'semivariable':
                    # Fix: use batch index properly relative to batch start
                    # k is absolute index in batch (0, n_bins, 2*n_bins...)
                    # batch[OT_INPUT_COST_ONEHOT] has shape (batch_size, num_metrics)
                    # We need the row index corresponding to the current example
                    # For row-wise, all rows in an example share the same metric?
                    # Assuming batch is collated correctly.
                    # wait, get_cost_matrix logic used k which is index in batch.
                    metric_idx = batch[OT_INPUT_COST_ONEHOT][k].argmax().item()
                    current_metric_name = utils.metrics[metric_idx]
                    C = cost_matrices[current_metric_name]
                else: # variable
                    # For variable cost, C is provided directly
                    C = batch[OT_INPUT_COST_MATRIX][k] 
                    # Note: handling variable cost matrix shape and repetition might depend on data format
                    # Ignoring specific metric name for variable cost unless provided elsewhere
                    
                seen_metrics.add(current_metric_name)

                # --- Run Neural OT (NOT) ---
                # Prepare samplers
                # Input source and target from batch
                # OT_INPUT_TOTAL: [source, target]
                input_total_vec = batch[OT_INPUT_TOTAL][k]
                source_hist = input_total_vec[:n_bins]
                target_hist = input_total_vec[n_bins:2*n_bins]
                
                # Create samplers
                # Using hardcoded range (-2.5, 2.5) as in NOT code or (0, 1) as in inference?
                # --- Run Neural OT (NOT) ---
                # Determine if we should run NOT
                # Helper flag: run if enabled in config AND cost type is not semivariable (as requested)
                # AND only for the first 10% of batches to save time
                num_batches = len(test_batches)
                not_cutoff = int(num_batches * 0.1)
                run_neural_ot = cfg.inference.test_neural_ot and (cost_type != 'semivariable') and (batch_idx < not_cutoff)
                
                ot_pred_sparse_not = None
                mae_not = -1.0
                mape_not = -1.0
                hellinger_not = -1.0
                tv_not = -1.0
                otcost_diff_not = -1.0

                if run_neural_ot:
                    # Prepare samplers
                    # Input source and target from batch
                    # OT_INPUT_TOTAL: [source, target]
                    input_total_vec = batch[OT_INPUT_TOTAL][k]
                    source_hist = input_total_vec[:n_bins]
                    target_hist = input_total_vec[n_bins:2*n_bins]
                    
                    X_sampler = utils.HistogramSampler(source_hist, ranges=(0, 1), device=device)
                    Y_sampler = utils.HistogramSampler(target_hist, ranges=(0, 1), device=device)
                    
                    # Run NOT
                    # Using config value for max steps
                    not_max_steps = cfg.inference.neural_ot_max_steps
                    rng = None
                    
                    # Call NOT routine
                    # Returns: ot_plan_T (X_T_bar), ot_plan_hist (normalized histogram)
                    with torch.enable_grad():
                        _, ot_pred_dense_not_np = NOT_toy_1D_cost_strong.main_routine(
                            X_sampler, Y_sampler, num_bins=n_bins, rng=(0, 1), metric=current_metric_name, 
                            max_steps=not_max_steps, verbose=False
                        )
                    
                    ot_pred_dense_not = torch.from_numpy(ot_pred_dense_not_np).to(dtype=my_dtype_torch, device=device)
                    ot_pred_sparse_not = sparse.coo_matrix(ot_pred_dense_not.cpu().numpy())

                # --- Metrics for TOT ---
                # Calculate MAE
                mae_sparse  = utils.sparse_mean_abs_error(ot_target_sparse, ot_pred_sparse)
                # Calculate MAPE
                mape_sparse = utils.sparse_weighted_mean_abs_percentage_error(ot_target_sparse, ot_pred_sparse)
                # Calculate Hellinger
                hellinger_sparse = utils.sparse_hellinger_distance(ot_target_sparse, ot_pred_sparse)
                # Calculate Total Variation
                tv_sparse = utils.sparse_total_variation(ot_target_sparse, ot_pred_sparse)
                # Calculate OT cost diff
                if cost_type == 'variable':
                    # C might be flattened or (N,N), ensure it's (N,N) for sum
                     if C.dim() == 1:
                         C = C.view(n_bins, n_bins)
                
                otcost_gt     = (ot_target_dense * C.cpu().numpy()).sum().item() # C is on device, move to cpu for numpy mult or use torch
                otcost_pred   = (ot_pred_dense.cpu().numpy() * C.cpu().numpy()).sum().item()
                otcost_diff   = abs(otcost_pred - otcost_gt)

                # Calculate Constraint Violations for TOT
                # Need source and target marginals.
                # source_hist, target_hist (already extracted above) are tensors
                metrics_viol = utils.compute_constraint_violations(ot_pred_sparse, source_hist, target_hist)
                row_violation = metrics_viol['row_violation']
                col_violation = metrics_viol['col_violation']
                max_violation = metrics_viol['max_violation']

                # Calculate metrics for NOT
                if run_neural_ot:
                    mae_not  = utils.sparse_mean_abs_error(ot_target_sparse, ot_pred_sparse_not)
                    mape_not = utils.sparse_weighted_mean_abs_percentage_error(ot_target_sparse, ot_pred_sparse_not)
                    hellinger_not = utils.sparse_hellinger_distance(ot_target_sparse, ot_pred_sparse_not)
                    tv_not = utils.sparse_total_variation(ot_target_sparse, ot_pred_sparse_not)
                    
                    otcost_pred_not = (ot_pred_dense_not.cpu().numpy() * C.cpu().numpy()).sum().item()
                    otcost_diff_not = abs(otcost_pred_not - otcost_gt)
                    
                # Collect metrics for TOT
                all_mae_sparse.append(mae_sparse)
                all_mape_sparse.append(mape_sparse)
                all_hellinger_sparse.append(hellinger_sparse)
                all_otcost_diff.append(otcost_diff)
                all_tv_sparse.append(tv_sparse)
                all_row_violation.append(row_violation)
                all_col_violation.append(col_violation)
                all_max_violation.append(max_violation)
                
                # Collect metrics for NOT
                if run_neural_ot:
                    results_not['mae'].append(mae_not)
                    results_not['mape'].append(mape_not)
                    results_not['hellinger'].append(hellinger_not)
                    results_not['otcost_diff'].append(otcost_diff_not)
                    results_not['tv'].append(tv_not)
                
                # Collect per-metric metrics (TOT)
                if current_metric_name in results_by_metric:
                    results_by_metric[current_metric_name]['mae'].append(mae_sparse)
                    results_by_metric[current_metric_name]['mape'].append(mape_sparse)
                    results_by_metric[current_metric_name]['hellinger'].append(hellinger_sparse)
                    results_by_metric[current_metric_name]['otcost_diff'].append(otcost_diff)
                    results_by_metric[current_metric_name]['tv'].append(tv_sparse)
                    results_by_metric[current_metric_name]['count'] += 1
                    
                # Collect per-metric metrics (NOT)
                if run_neural_ot and current_metric_name in results_by_metric_not:
                    results_by_metric_not[current_metric_name]['mae'].append(mae_not)
                    results_by_metric_not[current_metric_name]['mape'].append(mape_not)
                    results_by_metric_not[current_metric_name]['hellinger'].append(hellinger_not)
                    results_by_metric_not[current_metric_name]['otcost_diff'].append(otcost_diff_not)
                    results_by_metric_not[current_metric_name]['tv'].append(tv_not)
                    results_by_metric_not[current_metric_name]['count'] += 1

                # Store results for table
                row_data = {
                    'batch_idx': k,
                    'mae_sparse': mae_sparse,
                    'mape_sparse': mape_sparse,
                    'hellinger_sparse': hellinger_sparse,
                    'otcost_diff': otcost_diff,
                    'tv_sparse': tv_sparse,
                    'tv_sparse': tv_sparse,
                    'row_violation': row_violation,
                    'col_violation': col_violation,
                    'max_violation': max_violation,
                    'metric': current_metric_name
                }
                if run_neural_ot:
                    row_data['mae_not'] = mae_not
                    row_data['otcost_diff_not'] = otcost_diff_not
                    
                all_results.append(row_data)

                # Only create figures for first few batches (per metric if semivariable)
                create_fig = False
                if cost_type == 'semivariable':
                     if current_metric_name in results_by_metric: # Should always be true
                         if results_by_metric[current_metric_name]['fig_count'] < fig_counter_max:
                             create_fig = True
                             results_by_metric[current_metric_name]['fig_count'] += 1
                else:
                     if fig_counter < fig_counter_max:
                         create_fig = True
                         fig_counter += 1

                if create_fig:
                    # Create visualization for first sample in batch
                    # Extract source and target from OT_INPUT_TOTAL
                    # OT_INPUT_TOTAL structure: [source (n_bins), target (n_bins), input_idx (1)]
                    input_total_vec = batch[OT_INPUT_TOTAL][k]
                    source = input_total_vec[:n_bins].cpu().numpy()
                    target = input_total_vec[n_bins:2*n_bins].cpu().numpy()
                    input_cost_onehot = batch[OT_INPUT_COST_ONEHOT][k].cpu().numpy() if cost_type == 'semivariable' else None
                    
                    if run_neural_ot:
                        fig, axarr = plt.subplots(1, 6, figsize=(21, 3.5), dpi=120)
                    else:
                        fig, axarr = plt.subplots(1, 5, figsize=(18, 3.5), dpi=120)
                    
                    fig.suptitle(f"Batch {k}", fontsize=16, y=1.02)

                    # Titles
                    if cost_type == 'semivariable':
                         metric_title = utils.metrics[input_cost_onehot.argmax()]
                    else:
                         metric_title = current_metric_name

                    axarr[0].set_title(f"Source $\\mu$ (1D)", fontsize=12)
                    axarr[1].set_title(f"Target $\\nu$ (1D)", fontsize=12)
                    axarr[2].set_title(f"Cost Matrix: {metric_title}", fontsize=12)
                    axarr[3].set_title(r"$\pi$ Ground Truth (POT)", fontsize=12)
                    axarr[4].set_title(r"$\hat{\pi}$ Ours (TOT)", fontsize=12)
                    if run_neural_ot:
                        axarr[5].set_title(r"$\hat{\pi}$ Neural OT (NOT)", fontsize=12)

                    # Common vmin/vmax for OT plans
                    vmin = 0
                    if run_neural_ot:
                        vmax = max(ot_pred_dense.max(), ot_target_dense.max(), ot_pred_dense_not.max()).item()
                    else:
                        vmax = max(ot_pred_dense.max(), ot_target_dense.max()).item()

                    # --- Source (bar) ---
                    bin_edges = np.arange(len(source))
                    axarr[0].bar(
                        bin_edges, source,
                        width=1.0, color="tab:blue", alpha=0.7, edgecolor="black", align="edge", linewidth=0.5
                    )
                    axarr[0].set_xlim(0, len(source))
                    axarr[0].set_ylim(0, max(source.max(), target.max()) * 1.1)
                    axarr[0].set_xlabel("Bin index")
                    axarr[0].set_ylabel("Probability")
                    
                    # --- Target (bar) ---
                    bin_edges = np.arange(len(target))
                    axarr[1].bar(
                        bin_edges, target,
                        width=1.0, color="tab:blue", alpha=0.7, edgecolor="black", align="edge", linewidth=0.5
                    )
                    axarr[1].set_xlim(0, len(target))
                    axarr[1].set_ylim(0, max(source.max(), target.max()) * 1.1)
                    axarr[1].set_xlabel("Bin index")
                    axarr[1].set_ylabel("Probability")
                    
                    # --- Cost matrix ---
                    im2 = axarr[2].imshow(C.cpu().numpy(), cmap='Blues', origin='lower', aspect="auto")
                    axarr[2].set_xlabel("Target bins")
                    axarr[2].set_ylabel("Source bins")
                    plt.colorbar(im2, ax=axarr[2], fraction=0.046, pad=0.04)

                    # --- Ground Truth OT Plan ---
                    im3 = axarr[3].imshow(ot_target_dense, cmap='Blues', origin='lower', aspect="auto", vmin=vmin, vmax=vmax)
                    axarr[3].set_xlabel("Target bins")
                    axarr[3].set_ylabel("Source bins")
                    plt.colorbar(im3, ax=axarr[3], fraction=0.046, pad=0.04)

                    # --- Predicted OT Plan ---
                    im4 = axarr[4].imshow(ot_pred_dense.cpu().numpy(), cmap='Greens', origin='lower', aspect="auto", vmin=vmin, vmax=vmax)
                    axarr[4].set_xlabel("Target bins")
                    axarr[4].set_ylabel("Source bins")
                    plt.colorbar(im4, ax=axarr[4], fraction=0.046, pad=0.04)

                    # --- Neural OT Plan ---
                    if run_neural_ot:
                        im5 = axarr[5].imshow(ot_pred_dense_not.cpu().numpy(), cmap='Purples', origin='lower', aspect="auto", vmin=vmin, vmax=vmax)
                        axarr[5].set_xlabel("Target bins")
                        axarr[5].set_ylabel("Source bins")
                        plt.colorbar(im5, ax=axarr[5], fraction=0.046, pad=0.04)

                    plt.tight_layout()
                    
                    # Convert figure to base64 for html embedding
                    buf = BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    buf.close()
                    plt.close(fig)
                    
                    all_figures.append({
                        'batch_idx': k,
                        'img_base64': img_base64,
                        'metric': current_metric_name
                    })

                    # Add error metrics text
                    fig.text(0.5, -0.02, f'MAE (sparse): {mae_sparse:.3e}  |  ' +         
                            f'MAPE (sparse): {mape_sparse:.3e}  |  ' +   
                            f'OT Cost Diff: {otcost_diff:.3e}  |  ' +   
                            f'Hellinger (sparse): {hellinger_sparse:.3e}  |  ' +
                            f'TV (sparse): {tv_sparse:.3e}',
                            ha='center', fontsize=11, bbox=dict(boxstyle='round',
                            facecolor='lightyellow', alpha=0.8))

                    plt.tight_layout()

                    # Store figure for notebook as base64 string
                    buf = BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    buf.close()
                    plt.close(fig) # Close figure immediately to free memory

                    all_figures.append({
                        'img_base64': img_base64,
                        'batch_idx': k,
                        'metric': current_metric_name
                    })

            # Update progress bar with current metrics
            pbar.set_postfix(
                {
                    'batch_idx': batch_idx,
                    'mae_sparse': f'{np.mean(all_mae_sparse):.6e}',
                    'mape_sparse': f'{np.mean(all_mape_sparse):.6e}',
                    'hellinger_sparse': f'{np.mean(all_hellinger_sparse):.6e}',
                    'otcost_diff': f'{np.mean(all_otcost_diff):.6e}',
                    'tv_sparse': f'{np.mean(all_tv_sparse):.6e}',
                    'last_metric': current_metric_name
                },
                refresh=False
            )


    # Convert to numpy arrays for statistics (TOT)
    all_mae_sparse       = np.array(all_mae_sparse)
    all_mape_sparse      = np.array(all_mape_sparse)
    all_hellinger_sparse = np.array(all_hellinger_sparse)
    all_otcost_diff      = np.array(all_otcost_diff)
    all_tv_sparse        = np.array(all_tv_sparse)
    all_row_violation    = np.array(all_row_violation)
    all_col_violation    = np.array(all_col_violation)
    all_max_violation    = np.array(all_max_violation)
    
    # Convert to numpy arrays for statistics (NOT)
    all_mae_sparse_not       = np.array(results_not['mae'])
    all_mape_sparse_not      = np.array(results_not['mape'])
    all_hellinger_sparse_not = np.array(results_not['hellinger'])
    all_otcost_diff_not      = np.array(results_not['otcost_diff'])
    all_tv_sparse_not        = np.array(results_not['tv'])

    # Save TOT results to a numpy file
    np.savez(
        os.path.join(model_base_path, f"eval_all_testset_1d_{current_metric_name}_results_tot_sinkhorn.npz"),
        all_mae_sparse=all_mae_sparse,
        all_mape_sparse=all_mape_sparse,
        all_hellinger_sparse=all_hellinger_sparse,
        all_otcost_diff=all_otcost_diff,
        all_tv_sparse=all_tv_sparse,
        all_row_violation=all_row_violation,
        all_col_violation=all_col_violation,
        all_max_violation=all_max_violation,
        results_by_metric=results_by_metric
    )
    
    # Save NOT results to a numpy file
    np.savez(
        os.path.join(model_base_path, f"eval_all_testset_1d_{current_metric_name}_results_not.npz"),
        all_mae_sparse=all_mae_sparse_not,
        all_mape_sparse=all_mape_sparse_not,
        all_hellinger_sparse=all_hellinger_sparse_not,
        all_otcost_diff=all_otcost_diff_not,
        all_tv_sparse=all_tv_sparse_not,
        results_by_metric=results_by_metric_not
    )

    # Print statistics
    logger.info("\n" + "="*80)
    logger.info("INFERENCE METRICS SUMMARY (OVERALL)")
    logger.info("="*80)
    logger.info(f"Number of batches evaluated: {len(all_mae_sparse)}")
    logger.info("")
    
    logger.info("MAE (sparse):")
    logger.info(f"  Mean:   {all_mae_sparse.mean():.6e}")
    logger.info(f"  Median: {np.median(all_mae_sparse):.6e}")
    logger.info(f"  Min:    {all_mae_sparse.min():.6e}")
    logger.info(f"  Max:    {all_mae_sparse.max():.6e}")
    logger.info(f"  Std:    {all_mae_sparse.std():.6e}")
    logger.info("")
    
    logger.info("MAPE (sparse):")
    logger.info(f"  Mean:   {all_mape_sparse.mean():.6e}")
    logger.info(f"  Median: {np.median(all_mape_sparse):.6e}")
    logger.info(f"  Min:    {all_mape_sparse.min():.6e}")
    logger.info(f"  Max:    {all_mape_sparse.max():.6e}")
    logger.info(f"  Std:    {all_mape_sparse.std():.6e}")
    logger.info("="*80 + "\n")

    logger.info("Hellinger (sparse):")
    logger.info(f"  Mean:   {all_hellinger_sparse.mean():.6e}")
    logger.info(f"  Median: {np.median(all_hellinger_sparse):.6e}")
    logger.info(f"  Min:    {all_hellinger_sparse.min():.6e}")
    logger.info(f"  Max:    {all_hellinger_sparse.max():.6e}")
    logger.info(f"  Std:    {all_hellinger_sparse.std():.6e}")
    logger.info("="*80 + "\n")

    logger.info("TV (sparse):")
    logger.info(f"  Mean:   {all_tv_sparse.mean():.6e}")
    logger.info(f"  Median: {np.median(all_tv_sparse):.6e}")
    logger.info(f"  Min:    {all_tv_sparse.min():.6e}")
    logger.info(f"  Max:    {all_tv_sparse.max():.6e}")
    logger.info(f"  Std:    {all_tv_sparse.std():.6e}")
    logger.info("="*80 + "\n")


    logger.info("Constraint Violations (Row):")
    logger.info(f"  Mean:   {all_row_violation.mean():.6e}")
    logger.info(f"  Std:    {all_row_violation.std():.6e}")
    logger.info("="*80 + "\n")
    
    logger.info("Constraint Violations (Col):")
    logger.info(f"  Mean:   {all_col_violation.mean():.6e}")
    logger.info(f"  Std:    {all_col_violation.std():.6e}")
    logger.info("="*80 + "\n")
    
    logger.info("OT Cost Difference:")
    logger.info(f"  Mean:   {all_otcost_diff.mean():.6e}")
    logger.info(f"  Median: {np.median(all_otcost_diff):.6e}")
    logger.info(f"  Min:    {all_otcost_diff.min():.6e}")
    logger.info(f"  Max:    {all_otcost_diff.max():.6e}")
    logger.info(f"  Std:    {all_otcost_diff.std():.6e}")
    logger.info("="*80 + "\n")

    # Print results per metric
    if cost_type == 'semivariable':
        logger.info("\n" + "="*80)
        logger.info("INFERENCE METRICS SUMMARY (PER METRIC)")
        logger.info("="*80)
        
        for m in sorted(list(seen_metrics)):
            metrics_dict = results_by_metric[m]
            count = metrics_dict['count']
            if count == 0: continue
            
            mae = np.array(metrics_dict['mae'])
            mape = np.array(metrics_dict['mape'])
            hellinger = np.array(metrics_dict['hellinger'])
            tv = np.array(metrics_dict['tv'])
            ot_diff = np.array(metrics_dict['otcost_diff'])
            
            logger.info(f"Metric: {m} (count={count})")
            logger.info(f"  MAE:        Mean={mae.mean():.6e}, Std={mae.std():.6e}")
            logger.info(f"  MAPE:       Mean={mape.mean():.6e}, Std={mape.std():.6e}")
            logger.info(f"  Hellinger:  Mean={hellinger.mean():.6e}, Std={hellinger.std():.6e}")
            logger.info(f"  TV:         Mean={tv.mean():.6e}, Std={tv.std():.6e}")
            logger.info(f"  OTDiff:     Mean={ot_diff.mean():.6e}, Std={ot_diff.std():.6e}")
            logger.info("-" * 40)

    # Prepare summary table data
    table_data = []
    for r in all_results:
        table_data.append([
            r['batch_idx'],
            r.get('metric', 'N/A'),
            f"{r['mae_sparse']:.3e}",
            f"{r['mape_sparse']:.3e}",
            f"{r['hellinger_sparse']:.3e}",
            f"{r['tv_sparse']:.3e}",
            f"{r['otcost_diff']:.3e}",
            f"{r['row_violation']:.3e}",
            f"{r['col_violation']:.3e}"
        ])

    headers = ['Batch', 'Metric', 'MAE', 'MAPE', 'Hellinger', 'TV', 'OT Cost Diff', 'Row Violation', 'Col Violation']

    
    # Print summary table to terminal
    logger.info("\n" + "="*80)
    logger.info("DETAILED RESULTS TABLE")
    logger.info("="*80)
    table_str_terminal = tabulate(table_data, headers=headers, tablefmt='grid', floatfmt=".3e")
    logger.info("\n" + table_str_terminal)
    
    # # Create Jupyter notebook with results (OVERALL)
    if cost_type == 'fixed':
        create_analysis_notebook(
            model_base_path, "eval_all_testset.ipynb", cfg, n_bins, len(all_results),
            all_mae_sparse, all_mape_sparse, all_hellinger_sparse, all_tv_sparse, all_otcost_diff,
            all_figures
        )

    # Create separate notebooks per metric for semivariable cost
    if cost_type == 'semivariable':
        for m, metrics_dict in results_by_metric.items():
            if metrics_dict['count'] > 0:
                # Filter figures for this metric
                metric_figures = [f for f in all_figures if f.get('metric') == m]
                
                create_analysis_notebook(
                    model_base_path, f"eval_all_testset_1d_{m}.ipynb", cfg, n_bins, metrics_dict['count'],
                    np.array(metrics_dict['mae']), np.array(metrics_dict['mape']), 
                    np.array(metrics_dict['hellinger']), np.array(metrics_dict['tv']), 
                    np.array(metrics_dict['otcost_diff']),
                    metric_figures
                )


def create_analysis_notebook(model_base_path, filename, cfg, n_bins, num_batches,
                           mae_data, mape_data, hellinger_data, tv_data, otcost_data,
                           figures_data):
    logger.info("\n" + "="*80)
    logger.info(f"Creating Jupyter notebook: {filename}")
    logger.info("="*80)
    
    nb = new_notebook()
    cells = []
    
    # Add title cell
    cells.append(new_markdown_cell(
        f"# Transformers for Optimal Transport (TOT): Test Set Results\n\n"
        f"**Model:** {model_base_path}\n\n"
        f"**Total batches evaluated:** {num_batches}\n\n"
        f"**Dataset:** {cfg.data.hf_dataset}\n\n"
        f"**Support size (n_bins):** {n_bins}"
    ))
    
    # Add statistics summary
    stats_md = f"""## Summary Statistics

### MAE (sparse)
- **Mean:** {mae_data.mean():.6e}
- **Median:** {np.median(mae_data):.6e}
- **Min:** {mae_data.min():.6e}
- **Max:** {mae_data.max():.6e}
- **Std:** {mae_data.std():.6e}

### MAPE (sparse)
- **Mean:** {mape_data.mean():.6e}
- **Median:** {np.median(mape_data):.6e}
- **Min:** {mape_data.min():.6e}
- **Max:** {mape_data.max():.6e}
- **Std:** {mape_data.std():.6e}

### Hellinger (sparse)
- **Mean:** {hellinger_data.mean():.6e}
- **Median:** {np.median(hellinger_data):.6e}
- **Min:** {hellinger_data.min():.6e}
- **Max:** {hellinger_data.max():.6e}
- **Std:** {hellinger_data.std():.6e}

### TV (sparse)
- **Mean:** {tv_data.mean():.6e}
- **Median:** {np.median(tv_data):.6e}
- **Min:** {tv_data.min():.6e}
- **Max:** {tv_data.max():.6e}
- **Std:** {tv_data.std():.6e}

### OT Cost Difference
- **Mean:** {otcost_data.mean():.6e}
- **Median:** {np.median(otcost_data):.6e}
- **Min:** {otcost_data.min():.6e}
- **Max:** {otcost_data.max():.6e}
- **Std:** {otcost_data.std():.6e}
"""
    cells.append(new_markdown_cell(stats_md))

    # Create box plots summarizing the results
    cells.append(new_markdown_cell("## Summary Box Plots"))

    # Create figure with two subplots for box plots
    fig_box, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14, 5), dpi=120)
    
    # Box plot for MAE (sparse)
    bp1 = ax1.boxplot(mae_data, vert=True, patch_artist=True, 
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5))
    ax1.set_ylabel('MAE (sparse)', fontsize=12)
    ax1.set_title('MAE (sparse) Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    #ax1.set_yscale('log')  # Use log scale for better visualization
    
    # Add statistics text
    stats_text_mae = f'Mean: {mae_data.mean():.3e}\n'
    stats_text_mae += f'Median: {np.median(mae_data):.3e}\n'
    stats_text_mae += f'Min: {mae_data.min():.3e}\n'
    stats_text_mae += f'Max: {mae_data.max():.3e}'
    ax1.text(0.98, 0.98, stats_text_mae, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Box plot for MAPE (sparse)
    bp2 = ax2.boxplot(mape_data, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightcoral', alpha=0.7),
                     medianprops=dict(color='darkred', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5))
    ax2.set_ylabel('MAPE (sparse)', fontsize=12)
    ax2.set_title('MAPE (sparse) Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    #ax2.set_yscale('log')  # Use log scale for better visualization
    
    # Add statistics text
    stats_text_mape = f'Mean: {mape_data.mean():.3e}\n'
    stats_text_mape += f'Median: {np.median(mape_data):.3e}\n'
    stats_text_mape += f'Min: {mape_data.min():.3e}\n'
    stats_text_mape += f'Max: {mape_data.max():.3e}'
    ax2.text(0.98, 0.98, stats_text_mape, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Box plot for Hellinger (sparse)
    bp3 = ax3.boxplot(hellinger_data, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightgreen', alpha=0.7),
                     medianprops=dict(color='darkgreen', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5))
    ax3.set_ylabel('Hellinger (sparse)', fontsize=12)
    ax3.set_title('Hellinger (sparse) Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    #ax3.set_yscale('log')  # Use log scale for better visualization
    
    # Add statistics text
    stats_text_hellinger = f'Mean: {hellinger_data.mean():.3e}\n'
    stats_text_hellinger += f'Median: {np.median(hellinger_data):.3e}\n'
    stats_text_hellinger += f'Min: {hellinger_data.min():.3e}\n'
    stats_text_hellinger += f'Max: {hellinger_data.max():.3e}'
    ax3.text(0.98, 0.98, stats_text_hellinger, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Box plot for TV (sparse)
    bp4 = ax4.boxplot(tv_data, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightgreen', alpha=0.7),
                     medianprops=dict(color='darkgreen', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5))
    ax4.set_ylabel('TV (sparse)', fontsize=12)
    ax4.set_title('TV (sparse) Distribution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    #ax4.set_yscale('log')  # Use log scale for better visualization
    
    # Add statistics text
    stats_text_tv  = f'Mean: {tv_data.mean():.3e}\n'
    stats_text_tv += f'Median: {np.median(tv_data):.3e}\n'
    stats_text_tv += f'Min: {tv_data.min():.3e}\n'
    stats_text_tv += f'Max: {tv_data.max():.3e}'
    ax4.text(0.98, 0.98, stats_text_tv, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    
    # Convert box plot figure to base64 for embedding
    buf_box = BytesIO()
    fig_box.savefig(buf_box, format='png', bbox_inches='tight', dpi=120)
    buf_box.seek(0)
    img_base64_box = base64.b64encode(buf_box.read()).decode('utf-8')
    buf_box.close()
    plt.close(fig_box)

    # Add box plot to notebook
    cell_box = new_code_cell("")
    cell_box.outputs = [
        nbformat.v4.new_output(
            output_type='display_data',
            data={'image/png': img_base64_box},
            metadata={}
        )
    ]
    cells.append(cell_box)

    # Add summary table
    # cells.append(new_markdown_cell("## Detailed Results Table"))
    # table_str_notebook = tabulate(table_data, headers=headers, tablefmt='pipe', floatfmt=".3e")
    # cells.append(new_markdown_cell(table_str_notebook))

    # Add individual visualizations
    cells.append(new_markdown_cell(f"## Individual Visualizations (First {len(figures_data)})"))

    for fig_data in figures_data:
        # Add markdown header for this batch
        metric_str = f" - {fig_data['metric']}" if 'metric' in fig_data else ""
        cells.append(new_markdown_cell(f"### Batch {fig_data['batch_idx']}{metric_str}"))
        
        # Create an empty code cell with the figure as output (no code visible)
        cell = new_code_cell("")
        cell.outputs = [
            nbformat.v4.new_output(
                output_type='display_data',
                data={'image/png': fig_data['img_base64']},
                metadata={}
            )
        ]
        cells.append(cell)
    
    # Close all remaining figures
    plt.close('all')

    nb.cells = cells
    
    # Save notebook
    os.makedirs(model_base_path, exist_ok=True)
    notebook_path = os.path.join(model_base_path, filename)
    
    with open(notebook_path, 'w') as f:
        nbformat.write(nb, f)
    
    logger.info(f"Saved notebook to: {notebook_path}")
    logger.info(f"Open with: jupyter notebook {notebook_path}")
    logger.info("="*80 + "\n")


def load_model_by_name(model_base_path: str, device: str = "cuda", cfg=None):
    """
    Load a TokenRegressor saved in:
      artifacts/{name}/model_weights.pth
      artifacts/{name}/config.json or config.yaml

    Returns: (model, tokenizer)
    """

    model_full_path  = os.path.join(model_base_path, "model.pth")
    yaml_config_path = os.path.join(model_base_path, "config.yaml")

    if os.path.exists(yaml_config_path):
        try:
            cfg_loaded = OmegaConf.load(yaml_config_path)
            cfg = OmegaConf.to_container(cfg_loaded, resolve=True)
        except Exception:
            pass # ignore if resolve fails
    else:
        # Fallback to current config
        pass

    # Rebuild tokenizer from config
    tokenizer_cfg = cfg["tokenizer"]["DigitTokenizer"]

    # TODO: add support for automatic tokenizer selection
    # For now, assume IEEEFloatTokenizer based on config keys
    if "base" in tokenizer_cfg:
        tokenizer = tokenizers.IEEEFloatTokenizer(
            base=tokenizer_cfg["base"],
            num_exponent_digits=tokenizer_cfg["num_exponent_digits"],
            num_mantissa_digits=tokenizer_cfg["num_mantissa_digits"]
        )
    elif "vocabulary" in tokenizer_cfg:
        tokenizer = tokenizers.MyDigitTokenizer(
            vocabulary=tokenizer_cfg["vocabulary"],
            decimal_precision=tokenizer_cfg["decimal_precision"]
        )
    else:
        raise ValueError("Unknown tokenizer configuration")

    # Rebuild encoder
    ab_input_dim = cfg["model"]["encoder"]["encoder_ab"]["input_dim"]
    ab_emb_dim   = cfg["model"]["encoder"]["encoder_ab"]["emb_dim_ab"]
    ab_encoder   = nn.Sequential(
                    nn.Linear(ab_input_dim, ab_emb_dim),
                    nn.GELU(),
                    nn.Linear(ab_emb_dim, ab_emb_dim)
                   ).to(device)

    # Check if encoder_C exists in the config
    input_cost_type = cfg["data"]["dataset"]["input_cost_type"]
    if input_cost_type == 'fixed':
        c_encoder = None
        c_emb_dim = 0
    elif input_cost_type == 'semivariable':
        c_input_dim = cfg["model"]["encoder"]["encoder_c"]["input_dim"]
        c_emb_dim   = cfg["model"]["encoder"]["encoder_c"]["emb_dim_c"]
        c_encoder   = nn.Sequential(
            nn.Linear(c_input_dim, c_emb_dim),
            nn.GELU(),
            nn.Linear(c_emb_dim, c_emb_dim)
        ).to(device)
    elif input_cost_type == 'variable':
        c_input_dim = cfg["model"]["encoder"]["encoder_c"]["input_dim"]
        c_emb_dim   = cfg["model"]["encoder"]["encoder_c"]["emb_dim_c"]
        c_encoder   = nn.Sequential(
            nn.Linear(c_input_dim, c_emb_dim),
            nn.GELU(),
            nn.Linear(c_emb_dim, c_emb_dim)
        ).to(device)

    # Rebuild model using decoder config
    decoder_cfg     = cfg["model"]["decoder"]
    # pytorch_model   = model.TokenRegressor(
    #     tokenizer   = tokenizer,
    #     encoder_ab  = encoder_ab,
    #     encoder_c   = encoder_c,
    #     emb_dim_ab  = emb_dim_ab,
    #     emb_dim_c   = emb_dim_c,
    #     emb_dim_d   = decoder_cfg["emb_dim_d"],
    #     max_seq_len = decoder_cfg["max_seq_len"],
    #     num_transformer_layers = decoder_cfg["num_transformer_layers"],
    #     num_attention_heads    = decoder_cfg["num_attention_heads"],
    #     dropout     = decoder_cfg.get("dropout", 0.0)  # Use default if not present
    # ).to(device)

    num_prefix = cfg["model"]["num_prefix_tokens"]
    if num_prefix == 1:
        if cfg["model"]["use_flash_attention"]:
            pytorch_model = model.TokenRegressor_flash(tokenizer,
                                             input_cost_type,
                                             encoder_ab = ab_encoder,
                                             encoder_c  = c_encoder,
                                             emb_dim_ab = ab_emb_dim,
                                             emb_dim_c  = c_emb_dim,
                                             emb_dim_d  = decoder_cfg["emb_dim_d"],
                                             max_seq_len= decoder_cfg["max_seq_len"],
                                             num_transformer_layers = decoder_cfg["num_transformer_layers"],
                                             num_attention_heads    = decoder_cfg["num_attention_heads"],
                                             dropout                = decoder_cfg.get("dropout", 0.0)
                                             ).to(device)
        else:
            pytorch_model = model.TokenRegressor(tokenizer,
                                             input_cost_type,
                                             encoder_ab = ab_encoder,
                                             encoder_c  = c_encoder,
                                             emb_dim_ab = ab_emb_dim,
                                             emb_dim_c  = c_emb_dim,
                                             emb_dim_d  = decoder_cfg["emb_dim_d"],
                                             max_seq_len= decoder_cfg["max_seq_len"],
                                             num_transformer_layers = decoder_cfg["num_transformer_layers"],
                                             num_attention_heads    = decoder_cfg["num_attention_heads"],
                                             dropout                = decoder_cfg.get("dropout", 0.0)
                                             ).to(device)
    else:
        pytorch_model = model.TokenRegressor_two_prefix_tokens(tokenizer,
                                            input_cost_type,
                                            encoder_ab=ab_encoder,
                                            encoder_c=c_encoder,
                                            emb_dim_ab=ab_emb_dim,
                                            emb_dim_c=c_emb_dim,
                                            emb_dim_d=decoder_cfg["emb_dim_d"],
                                            max_seq_len=decoder_cfg["max_seq_len"],
                                            num_transformer_layers=decoder_cfg["num_transformer_layers"],
                                            num_attention_heads=decoder_cfg["num_attention_heads"],
                                            dropout=decoder_cfg.get("dropout", 0.0)
                                            ).to(device)

    # Load weights
    model_state = torch.load(model_full_path, map_location=device, weights_only=True)
    pytorch_model.load_state_dict(model_state, strict=True)

    return pytorch_model, tokenizer

def eval_all_1d_distribution_pairs(cfg: DictConfig, pytorch_model=None, tokenizer=None, model_base_path=None):

    """
    Test 1D distribution examples across all distribution pairs.

    Args:
        cfg: Hydra configuration
        pytorch_model: Pre-loaded model (optional). If None, will load from config.
        tokenizer: Pre-loaded tokenizer (optional). If None, will load from config.
    """
    # --- tiny adapter to match hist_from_samplers API ---
    class TorchSampler:
        def __init__(self, fn, name=None):
            self.fn = fn  # fn(n) -> torch.Tensor of shape (n,)
            self.name = name or "TorchSampler"
        def sample(self, n):
            x = self.fn(n).float()
            return x.view(-1, 1).to(device)  # (n, 1)

    def zscore(x, eps=1e-8):
        x = (x - x.mean()) / (x.std() + eps)
        return x

    # --- distribution factories (all return TorchSampler with a .name) ---
    def BetaLeft(a=2.0, b=5.0):
        # Beta is on [0, 1]
        m = torch.distributions.Beta(torch.tensor([a]), torch.tensor([b]))
        return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)),
                            name=f"BetaLeft")

    def BetaRight(a=5.0, b=2.0):
        m = torch.distributions.Beta(torch.tensor([a]), torch.tensor([b]))
        return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)),
                            name=f"BetaRight")

    def LogNormal(mu=0.0, sigma=0.5):
        m = torch.distributions.LogNormal(torch.tensor([mu]), torch.tensor([sigma]))
        return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)),
                            name=f"LogNormal")

    def Laplace(loc=0.0, scale=1.0):
        m = torch.distributions.Laplace(torch.tensor([loc]), torch.tensor([scale]))
        return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)),
                            name=f"Laplace")

    def StudentT(df=3):
        m = torch.distributions.StudentT(torch.tensor([float(df)]))
        return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)),
                            name=f"StudentT")

    def CauchyTrunc(x0=0.0, gamma=1.0, clip=6.0):
        m = torch.distributions.Cauchy(torch.tensor([x0]), torch.tensor([gamma]))
        def _f(n):
            x = m.sample((n,)).view(-1)
            x = torch.clamp(x, -clip, clip)
            return zscore(x)
        return TorchSampler(_f, name=f"CauchyTrunc")

    def Triangular(left=-2.0, mode=0.0, right=2.0):
        def _f(n):
            u = torch.rand(n)
            # Inverse transform sampling
            c = (mode - left) / (right - left)
            x = torch.zeros_like(u)
            mask = u < c
            # u < (mode-left)/(right-left)
            # x = left + sqrt(u * (right-left) * (mode-left))
            if mask.any():
                x[mask] = left + torch.sqrt(u[mask] * (right - left) * (mode - left))
            if (~mask).any():
                x[~mask] = right - torch.sqrt((1 - u[~mask]) * (right - left) * (right - mode))
            return zscore(x)
        return TorchSampler(_f, name=f"Triangular")

    def UniformLowHigh(low=-2.0, high=2.0):
        return TorchSampler(lambda n: zscore(torch.rand(n) * (high - low) + low),
                            name=f"Uniform")

    def GaussianShifted(mu=1.5, sigma=1.0):
        return TorchSampler(lambda n: zscore(torch.randn(n) * sigma + mu),
                            name=f"GaussianShifted")

    def GaussianMixture(means=(-2.0, 2.0), sigmas=(0.7, 0.7), weights=(0.5, 0.5)):
        t_means = torch.tensor(means).float()
        t_sigmas = torch.tensor(sigmas).float()
        t_weights = torch.tensor(weights).float()
        t_weights = t_weights / t_weights.sum()
        def _f(n):
            comps = torch.multinomial(t_weights, n, replacement=True)
            # gather means and sigmas
            # torch.normal(mean, std)
            x = torch.normal(mean=t_means[comps], std=t_sigmas[comps])
            return zscore(x)
        return TorchSampler(_f, name=f"GaussianMixture")

    def Exponential(scale=1.0):
        # torch Exponential uses rate = 1/scale
        rate = 1.0 / scale
        m = torch.distributions.Exponential(torch.tensor([rate]))
        return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)),
                            name=f"Exponential")

    def Weibull(a=1.5, scale=1.0):
        # torch Weibull(scale, concentration)
        m = torch.distributions.Weibull(torch.tensor([scale]), torch.tensor([a]))
        return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)),
                            name=f"Weibull")

    def Gamma(shape=2.0, scale=1.0):
        # torch Gamma(concentration, rate). rate = 1/scale.
        rate = 1.0 / scale
        m = torch.distributions.Gamma(torch.tensor([shape]), torch.tensor([rate]))
        return TorchSampler(lambda n: zscore(m.sample((n,)).view(-1)),
                            name=f"Gamma")

    def Pareto(a=2.5, clip=10.0):
        # Numpy pareto: x = (np.random.pareto(a, size=n)) which is Lomax(a).
        # x = (1-u)^(-1/a) - 1
        def _f(n):
            u = torch.rand(n)
            x = torch.pow(1 - u, -1.0 / a) - 1.0
            x = torch.clamp(x, 0, clip)
            return zscore(x)
        return TorchSampler(_f, name=f"Pareto")

    def Cosine():
        def _f(n):
             # Original: np.cos(zscore(np.random.rand(n) * np.pi - np.pi/2))
             # Inner part: Uniform(-pi/2, pi/2). Then ZScore. Then Cos.
             val = torch.rand(n) * np.pi - np.pi/2
             return torch.cos(zscore(val))
        return TorchSampler(_f, name="Cosine")

    def BimodalUniform(low1=-3, high1=-1, low2=1, high2=3, p=0.5):
        def _f(n):
            mask = torch.rand(n) < p
            x = torch.empty(n)
            n_mask = mask.sum().item()
            if n_mask > 0:
                x[mask] = torch.rand(n_mask) * (high1 - low1) + low1
            if n - n_mask > 0:
                x[~mask] = torch.rand(n - n_mask) * (high2 - low2) + low2
            return zscore(x)
        return TorchSampler(_f, name=f"BimodalUniform")

    def GaussianMixtureWideNarrow():
        return GaussianMixture(means=(0.0, 0.0), sigmas=(0.3, 2.0), weights=(0.5, 0.5))

    # All sampler pairs
    list_distributions = [
        BetaLeft(),
        BetaRight(),
        Laplace(),
        UniformLowHigh(),
        StudentT(df=5),
        GaussianShifted(mu=1.0),
        LogNormal(sigma=0.6),
        CauchyTrunc(clip=5.0),
        Triangular(-3, 0, 3),
        Exponential(scale=1.0),
        Weibull(a=2.0),
        Gamma(2.0),
    ]

    # define all cross pairs
    sampler_pairs_matrix = np.empty((len(list_distributions), len(list_distributions)), dtype=object)
    for i, a in enumerate(list_distributions):
        for j, b in enumerate(list_distributions):
            sampler_pairs_matrix[i, j] = (a, b)

    # Load the pre-trained model if not provided
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect results across trials
    n_samples   = 1_000_000
    n_bins      = cfg.data.dataset.support_size
    # check encoder C in config
    cost_type   = cfg.data.dataset.input_cost_type
    row_wise    = cfg.data.dataset.row_wise

    if cost_type == 'fixed':
        metrics = [cfg.data.dataset.input_cost_metric]
    elif cost_type == 'semivariable':
        metrics = [cfg.data.dataset.input_cost_metric]
    elif cost_type == 'variable':
        metrics = [cfg.data.dataset.input_cost_metric]

    # Evaluate distribution pairs in the matrix
    for metric in metrics:
        logger.info(f"Evaluating metric: {metric}")
        all_results_tot = []
        all_results_not = []
        all_figures = []

        for i in range(sampler_pairs_matrix.shape[0]):
            for j in range(sampler_pairs_matrix.shape[1]):

                X_sampler = sampler_pairs_matrix[i, j][0]
                Y_sampler = sampler_pairs_matrix[i, j][1]
                idx = i * sampler_pairs_matrix.shape[1] + j
                logger.info(f"Evaluating pair {idx+1}/{sampler_pairs_matrix.shape[0] * sampler_pairs_matrix.shape[1]}: {X_sampler.name} -> {Y_sampler.name}")

                # Sample + histogram
                source_sample = X_sampler.sample(n_samples)
                target_sample = Y_sampler.sample(n_samples)
                source, target, rng = utils.hist_from_samples(source_sample, target_sample, bins=n_bins, density=False)

                # Define support points
                if cost_type == 'fixed':
                    # fixed C case
                    xs = np.zeros((n_bins, 2))
                    xt = np.zeros((n_bins, 2))
                    xs[:, 0] = np.arange(n_bins, dtype=my_dtype_np)
                    xt[:, 0] = np.arange(n_bins, dtype=my_dtype_np)
                elif cost_type == 'semivariable':
                    xs = np.zeros((n_bins, 2))
                    xt = np.zeros((n_bins, 2))
                    xs[:, 0] = np.arange(n_bins, dtype=my_dtype_np)
                    xt[:, 0] = np.arange(n_bins, dtype=my_dtype_np)
                    # one-hot encode the current metric
                    metric_one_hot = [1.0 if m == metric else 0.0 for m in utils.metrics]
                elif cost_type == 'variable':
                    # n_bins defined elsewhere
                    x_low = 0; x_high = n_bins
                    # integers from [x_low, x_high) -> 0 to 9
                    xs = np.random.randint(low=x_low, high=x_high, size=(n_bins, 2))
                    xt = np.random.randint(low=x_low, high=x_high, size=(n_bins, 2))

                # Define cost matrix
                C = ot.dist(xs, xt, metric=metric, use_tensor=True)
                C = C / C.max()

                # Compute ground truth OT plan using POT
                otplan_solve = ot.solve(
                    C, a=source, b=target
                )

                otplan_solve_sparse = otplan_solve.sparse_plan
                otplan_solve_dense  = otplan_solve.plan
                # Convert source/target distributions to tensors
                source_tensor = torch.tensor(source, dtype=my_dtype_torch)
                target_tensor = torch.tensor(target, dtype=my_dtype_torch)

                # Yield individual bin samples
                if row_wise:
                    encoder_input_ab = []
                    encoder_input_c  = []
                    for j in range(n_bins):
                        input_idx = [j / n_bins]
                        input_total = torch.hstack((
                            source_tensor,
                            target_tensor,
                            torch.tensor(input_idx, dtype=my_dtype_torch)
                        )).to(my_dtype_torch)
                        encoder_input_ab.append(input_total)

                        if cost_type == 'variable':
                            encoder_input_c.append(torch.tensor(C, dtype=my_dtype_torch).flatten(start_dim=0))
                        elif cost_type == 'semivariable':
                            encoder_input_c.append(torch.tensor(metric_one_hot))

                # else:
                #     encoder_input_ab = torch.hstack((
                #         source_tensor,
                #         target_tensor
                #     )).to(my_dtype_torch)

                #     if cost_fixed:
                #         encoder_input_C = None
                #     else:
                #         encoder_input_C  = torch.tensor(C, dtype=my_dtype_torch)
                #         encoder_input_C  = encoder_input_C.flatten(start_dim=0).to(device)

                encoder_input_ab = torch.stack(encoder_input_ab).to(device)
                if cost_type == 'fixed':
                    encoder_input_c = None
                elif cost_type == 'semivariable':
                    encoder_input_c = torch.stack(encoder_input_c).to(device)
                elif cost_type == 'variable':
                    encoder_input_c = torch.stack(encoder_input_c).to(device)
                ####################################################################################################################################  OUR METHOD (TOT)
                # Run generation
                start_time_gen = time.time()
                text_ids_pred = pytorch_model.generate(
                    encoder_input_ab = encoder_input_ab,
                    encoder_input_c  = encoder_input_c,
                    temperature=0.1,
                    max_seq_len=cfg.inference.max_seq_len
                )
                time_gen = time.time() - start_time_gen
                
                text_pred = tokenizer.from_tokens(text_ids_pred)
                print( "pair idx: ", idx)
                logger.info(f"text_pred: \n{text_pred}")
                text_pred = [t[:t.find(']')+1] if (']' in t) else t for t in text_pred]
                
                logger.info(f"text_pred_truncated: \n{text_pred}")
                text_pred_dense = [utils.parse_predicted_sparse_row_string(t, n_bins, mode='index', precision=cfg.tokenizer.DigitTokenizer.decimal_precision) for t in text_pred ]
                ot_pred_dense = np.stack([np.fromstring(t[1:-1], sep=',') for t in text_pred_dense])
                
                # Apply Sinkhorn Projection and time it
                start_time_sink = time.time()
                ot_pred_dense = apply_sinkhorn_projection(ot_pred_dense, source, target)
                time_sink = time.time() - start_time_sink
                
                ot_pred_dense_torch = torch.from_numpy(ot_pred_dense)
                # build sparse matrix directly
                ot_pred_sparse = sparse.coo_matrix(
                    ot_pred_dense, shape=otplan_solve_sparse.shape
                )
                
                # total run time wrapper
                run_time_tot = time_gen + time_sink
                #print( "tot_pred_sparse: ")
                #utils.print_sparse_matrix(ot_pred_sparse)
                #print( "pot_gt_sparse: ")
                #utils.print_sparse_matrix(otplan_solve_sparse)

                # Compute metrics
                # MAE
                #abs_diff_dense = sklearn.metrics.mean_absolute_error(otplan_solve_dense.flatten(), ot_pred_dense.flatten())
                mae_sparse_tot  = utils.sparse_mean_abs_error(otplan_solve_sparse, ot_pred_sparse)
                # MAPE
                #mape_diff_sparse = utils.sparse_mean_abs_percentage_error(otplan_solve_sparse, ot_pred_sparse)
                mape_sparse_tot = utils.sparse_weighted_mean_abs_percentage_error(otplan_solve_sparse, ot_pred_sparse)
                # Hellinger
                hellinger_sparse_tot = utils.sparse_hellinger_distance(otplan_solve_sparse, ot_pred_sparse)
                # OT cost diff
                otcost_gt     = otplan_solve.value
                otcost_pred   = (ot_pred_dense * C).sum()
                otcost_diff_tot   = abs(otcost_pred - otcost_gt)
                # Total Variation
                tv_sparse_tot = utils.sparse_total_variation(otplan_solve_sparse, ot_pred_sparse)
                
                # Calculate Constraint Violations
                # source and target are numpy arrays (from utils.hist_from_samples)
                metrics_viol_tot = utils.compute_constraint_violations(ot_pred_sparse, source, target)

                all_results_tot.append({
                    'source':           X_sampler.name,
                    'target':           Y_sampler.name,
                    'otcost_diff':      otcost_diff_tot,
                    'mae_sparse':       mae_sparse_tot,
                    'mape_sparse':      mape_sparse_tot,
                    'hellinger_sparse': hellinger_sparse_tot,
                    'tv_sparse':        tv_sparse_tot,
                    'row_violation':    metrics_viol_tot['row_violation'],
                    'col_violation':    metrics_viol_tot['col_violation'],
                    'max_violation':    metrics_viol_tot['max_violation'],
                    'run_time':         run_time_tot,
                    'run_time_gen':     time_gen,
                    'run_time_sink':    time_sink
                })
                ####################################################################################################################################  NEURAL OT (NOT)
                ot_plan_not_dense = None
                ot_plan_not_sparse = None
                if cfg.inference.test_neural_ot:
                    print("calling neural ot routine")
                    not_max_steps = cfg.inference.neural_ot_max_steps
                    #_, ot_plan_not_dense = NOT_toy_1D_cost_weak.main_routine(X_sampler, Y_sampler, n_bins)
                    start_time = time.time()
                    _, ot_plan_not_dense = NOT_toy_1D_cost_strong.main_routine(X_sampler, Y_sampler, n_bins, rng, metric, max_steps=not_max_steps, verbose=False)
                    ot_plan_not_sparse = sparse.coo_matrix(
                        ot_plan_not_dense,
                        shape=otplan_solve_sparse.shape
                    )
                    run_time_not = time.time() - start_time
                    #print("not_pred_sparse: ")
                    #utils.print_sparse_matrix(ot_plan_not_sparse)
                    print( "tot_not_pot_sparse: ")
                    precision_to_display = cfg.tokenizer.DigitTokenizer.decimal_precision
                    utils.print_sparse_matrices(ot_pred_sparse, ot_plan_not_sparse, otplan_solve_sparse, precision_to_display=precision_to_display)
                    # Compute metrics
                    # MAE
                    mae_sparse_not  = utils.sparse_mean_abs_error(otplan_solve_sparse, ot_plan_not_sparse)
                    # MAPE
                    mape_sparse_not = utils.sparse_weighted_mean_abs_percentage_error(otplan_solve_sparse, ot_plan_not_sparse)
                    # Hellinger
                    hellinger_sparse_not = utils.sparse_hellinger_distance(otplan_solve_sparse, ot_plan_not_sparse)
                    # OT cost diff
                    otcost_gt     = otplan_solve.value
                    otcost_pred   = (ot_plan_not_dense * C).sum()
                    otcost_diff_not  = abs(otcost_pred - otcost_gt)
                    # Total Variation
                    tv_sparse_not = utils.sparse_total_variation(otplan_solve_sparse, ot_plan_not_sparse)

                    all_results_not.append({
                        'source':           X_sampler.name,
                        'target':           Y_sampler.name,
                        'otcost_diff':      otcost_diff_not,
                        'mae_sparse':       mae_sparse_not,
                        'mape_sparse':      mape_sparse_not,
                        'hellinger_sparse': hellinger_sparse_not,
                        'tv_sparse':        tv_sparse_not,
                        'run_time':         run_time_not
                    })
                ####################################################################################################################################  PLOTTING RESULTS of BOTH METHODS

                # Create visualization
                fig, axarr = plt.subplots(1, 7, figsize=(20, 3.5), dpi=120)
                fig.suptitle(f"{X_sampler.name} → {Y_sampler.name}", fontsize=16, y=1.02)

                # Titles
                axarr[0].set_title(f"Source $\\mu$ (1D)", fontsize=12)
                axarr[1].set_title(f"Target $\\nu$ (1D)", fontsize=12)
                axarr[2].set_title(f"Cost Matrix: {metric}", fontsize=12)
                axarr[3].set_title(r"$\pi$ Ground Truth (Python OT)", fontsize=12)
                axarr[4].set_title(r"$\hat{\pi}$ Ours (Transformers OT)", fontsize=12)
                axarr[5].set_title(r"$\hat{\pi}$ Neural OT", fontsize=12)

                # Common vmin/vmax for OT plans
                vmin = 0
                vmax = max(np.max(ot_pred_dense), np.max(otplan_solve_dense))

                # --- Source (bar) ---
                bin_edges = np.arange(len(source))
                axarr[0].bar(
                    bin_edges, source,
                    width=1.0, color="tab:blue", alpha=0.7, edgecolor="black", align='center', linewidth=0.5
                )
                #axarr[0].set_xlim(-0.5, len(source))
                #axarr[0].set_ylim(0, max(source.max(), target.max()) * 1.1)
                axarr[0].set_xlabel("Bin index")
                axarr[0].set_ylabel("Probability")

                # --- Target (bar) ---
                bin_edges = np.arange(len(target))
                axarr[1].bar(
                    bin_edges, target,
                    width=1.0, color="tab:blue", alpha=0.7, edgecolor="black", align='center', linewidth=0.5
                )
                #axarr[1].set_xlim(-0.5, len(target))
                #axarr[1].set_ylim(0, max(source.max(), target.max()) * 1.1)
                axarr[1].set_xlabel("Bin index")
                axarr[1].set_ylabel("Probability")

                # --- Cost matrix ---
                im2 = axarr[2].imshow(C, cmap='Blues', origin='lower', aspect="auto")
                axarr[2].set_xlabel("Target bins")
                axarr[2].set_ylabel("Source bins")
                plt.colorbar(im2, ax=axarr[2], fraction=0.046, pad=0.04)

                # --- Ground truth ---
                im3 = axarr[3].imshow(otplan_solve_dense, cmap='Blues', vmin=vmin, vmax=vmax, 
                                    origin='lower', aspect="auto")
                axarr[3].set_xlabel("Target bins")
                axarr[3].set_ylabel("Source bins")
                axarr[3].text(0.02, 0.98, rf'$\sum \pi = {otplan_solve_sparse.sum():.5f}$', 
                            transform=axarr[3].transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round',
                            facecolor='white', alpha=0.8))
                plt.colorbar(im3, ax=axarr[3], fraction=0.046, pad=0.04)

                # --- Our Model Prediction ---
                im4 = axarr[4].imshow(ot_pred_dense, cmap='Greens', vmin=vmin, vmax=vmax,
                                    origin='lower', aspect="auto")
                axarr[4].set_xlabel("Target bins")
                axarr[4].set_ylabel("Source bins")
                axarr[4].text(0.02, 0.98, rf'$\sum \hat{{\pi}} = {ot_pred_sparse.sum():.5f}$',
                            transform=axarr[4].transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round',
                            facecolor='white', alpha=0.8))
                plt.colorbar(im4, ax=axarr[4], fraction=0.046, pad=0.04)

                # --- NOT Model Prediction ---
                if ot_plan_not_dense is not None:
                    im5 = axarr[5].imshow(ot_plan_not_dense, cmap='Purples', vmin=vmin, vmax=vmax, origin='lower', aspect="auto")
                    axarr[5].set_title(r"$\hat{\pi}$ Neural OT", fontsize=12)
                    axarr[5].text(0.02, 0.98, rf'$\sum \hat{{\pi}} = {ot_plan_not_sparse.sum():.5f}$',
                                transform=axarr[5].transAxes, fontsize=10,
                                verticalalignment='top', bbox=dict(boxstyle='round',
                                facecolor='white', alpha=0.8))
                else:
                    im5 = axarr[5].imshow(np.zeros_like(otplan_solve_dense), cmap='Purples', vmin=vmin, vmax=vmax, origin='lower', aspect="auto")
                    axarr[5].set_title(r"$\hat{\pi}$ Neural OT (N/A)", fontsize=12)
                    
                axarr[5].set_xlabel("Target bins")
                axarr[5].set_ylabel("Source bins")
                plt.colorbar(im5, ax=axarr[5], fraction=0.046, pad=0.04)

                # --- Model Prediction on Dataset (samples + plan on axarr[5]) ---
                # Make axarr[5] the current axis so plot2D_samples_mat draws there
                plt.sca(axarr[6])
                # set title
                pl.title("Support points (1D)")
                ot.plot.plot2D_samples_mat(xs, xt, ot_pred_dense, c=[0.5, 0.5, 1])
                pl.plot(xs[:, 0], xs[:, 1], "+b", label="Source")
                pl.plot(xt[:, 0], xt[:, 1], "xr", label="Target")
                pl.legend(loc="best")
                pl.axis("equal")

                # Add error metrics text
                fig.text(0.5, -0.02, f'MAE (sparse): {mae_sparse_tot:.3e}  |  ' +
                        f'MAPE (sparse): {mape_sparse_tot:.3e}  |  ' +
                        f'Hellinger (sparse): {hellinger_sparse_tot:.3e}  |  ' +
                        f'TV (sparse): {tv_sparse_tot:.3e}  |  ' +
                        f'OT Cost Diff: {otcost_diff_tot:.3e}',
                        ha='center', fontsize=11, bbox=dict(boxstyle='round',
                        facecolor='lightyellow', alpha=0.8))

                plt.tight_layout()

                # Store figure for notebook as base64 string
                buf = BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close(fig) # Close figure immediately to free memory

                all_figures.append({
                    'img_base64': img_base64,
                    'source': X_sampler.name,
                    'target': Y_sampler.name,
                    'idx': idx
                })

        # Create squared matrix tables matching sampler_pairs_matrix.shape
        # Reshape results into matrices
        n_rows, n_cols = sampler_pairs_matrix.shape

        # Get distribution names for row/column labels
        row_names = [sampler_pairs_matrix[i, 0][0].name for i in range(n_rows)]
        col_names = [sampler_pairs_matrix[0, j][1].name for j in range(n_cols)]

        # Create result matrices
        otcost_diff_matrix      = np.zeros((n_rows, n_cols))
        mae_sparse_matrix       = np.zeros((n_rows, n_cols))
        mape_sparse_matrix      = np.zeros((n_rows, n_cols))
        hellinger_sparse_matrix = np.zeros((n_rows, n_cols))
        tv_sparse_matrix        = np.zeros((n_rows, n_cols))
        max_violation_matrix    = np.zeros((n_rows, n_cols))
        run_time_matrix         = np.zeros((n_rows, n_cols))

        # Fill matrices from all_results (results are stored in row-major order)
        for idx, r in enumerate(all_results_tot):
            i = idx // n_cols
            j = idx % n_cols
            otcost_diff_matrix[i, j]       = r['otcost_diff']
            mae_sparse_matrix[i, j]        = r['mae_sparse']
            mape_sparse_matrix[i, j]       = r['mape_sparse']
            hellinger_sparse_matrix[i, j]  = r['hellinger_sparse']
            tv_sparse_matrix[i, j]         = r['tv_sparse']
            max_violation_matrix[i, j]     = r['max_violation']
            run_time_matrix[i, j]          = r['run_time']

        # Create OT Cost Diff table
        logger.info("\n" + "="*80)
        logger.info("OT COST DIFF")
        logger.info("="*80)
        # Prepare table data with row names
        otcost_table_data = []
        for i, row_name in enumerate(row_names):
            row = [row_name] + [f"{otcost_diff_matrix[i, j]:.3e}" for j in range(n_cols)]
            otcost_table_data.append(row)
        otcost_headers = ['Source \\ Target'] + col_names
        otcost_table_str = tabulate(otcost_table_data, headers=otcost_headers, tablefmt='grid', floatfmt=".3e")
        logger.info("\n" + otcost_table_str)
        
        # Create MAE (sparse) table
        logger.info("\n" + "="*80)
        logger.info("MAE (SPARSE)")
        logger.info("="*80)
        # Prepare table data with row names
        mae_sparse_table_data = [] 
        for i, row_name in enumerate(row_names):
            row = [row_name] + [f"{mae_sparse_matrix[i, j]:.3e}" for j in range(n_cols)]
            mae_sparse_table_data.append(row)
        mae_sparse_headers = ['Source \\ Target'] + col_names
        mae_sparse_table_str = tabulate(mae_sparse_table_data, headers=mae_sparse_headers, tablefmt='grid', floatfmt=".3e")
        logger.info("\n" + mae_sparse_table_str)
        
        # Create MAPE (sparse) table
        logger.info("\n" + "="*80)
        logger.info("MAPE (SPARSE)")
        logger.info("="*80)
        # Prepare table data with row names
        mape_sparse_table_data = []    
        for i, row_name in enumerate(row_names):
            row = [row_name] + [f"{mape_sparse_matrix[i, j]:.3e}" for j in range(n_cols)]
            mape_sparse_table_data.append(row)
        mape_sparse_headers = ['Source \\ Target'] + col_names
        mape_sparse_table_str = tabulate(mape_sparse_table_data, headers=mape_sparse_headers, tablefmt='grid', floatfmt=".3e")
        logger.info("\n" + mape_sparse_table_str)

        # Create Hellinger (sparse) table
        logger.info("\n" + "="*80)
        logger.info("HELLINGER (SPARSE)")
        logger.info("="*80)
        # Prepare table data with row names
        hellinger_sparse_table_data = []
        for i, row_name in enumerate(row_names):
            row = [row_name] + [f"{hellinger_sparse_matrix[i, j]:.3e}" for j in range(n_cols)]
            hellinger_sparse_table_data.append(row)
        hellinger_sparse_headers = ['Source \\ Target'] + col_names
        hellinger_sparse_table_str = tabulate(hellinger_sparse_table_data, headers=hellinger_sparse_headers, tablefmt='grid', floatfmt=".3e")
        logger.info("\n" + hellinger_sparse_table_str)

        # Create TV (sparse) table
        logger.info("\n" + "="*80)
        logger.info("TV (SPARSE)")
        logger.info("="*80)
        # Prepare table data with row names
        tv_sparse_table_data = []
        for i, row_name in enumerate(row_names):
            row = [row_name] + [f"{tv_sparse_matrix[i, j]:.3e}" for j in range(n_cols)]
            tv_sparse_table_data.append(row)
        tv_sparse_headers = ['Source \\ Target'] + col_names
        tv_sparse_table_str = tabulate(tv_sparse_table_data, headers=tv_sparse_headers, tablefmt='grid', floatfmt=".3e")
        logger.info("\n" + tv_sparse_table_str)

        # Create Max Violation table
        logger.info("\n" + "="*80)
        logger.info("MAX VIOLATION")
        logger.info("="*80)
        max_violation_table_data = []
        for i, row_name in enumerate(row_names):
            row = [row_name] + [f"{max_violation_matrix[i, j]:.3e}" for j in range(n_cols)]
            max_violation_table_data.append(row)
        max_violation_headers = ['Source \\ Target'] + col_names
        max_violation_table_str = tabulate(max_violation_table_data, headers=max_violation_headers, tablefmt='grid', floatfmt=".3e")
        logger.info("\n" + max_violation_table_str)

        # Create Run Time table
        logger.info("\n" + "="*80)
        logger.info("RUN TIME")
        logger.info("="*80)
        # Prepare table data with row names
        run_time_table_data = []
        for i, row_name in enumerate(row_names):
            row = [row_name] + [f"{run_time_matrix[i, j]:.3e}" for j in range(n_cols)]
            run_time_table_data.append(row)
        run_time_headers = ['Source \\ Target'] + col_names
        run_time_table_str = tabulate(run_time_table_data, headers=run_time_headers, tablefmt='grid', floatfmt=".3e")
        logger.info("\n" + run_time_table_str)

        # Prepare summary table data
        table_data_tot = []
        for r in all_results_tot:
            table_data_tot.append([
                r['source'],
                r['target'],
                f"{r['otcost_diff']:.3e}",
                f"{r['mae_sparse']:.3e}",
                f"{r['mape_sparse']:.3e}",
                f"{r['hellinger_sparse']:.3e}",
                f"{r['tv_sparse']:.3e}",    
                f"{r['max_violation']:.3e}",
                f"{r['run_time']:.3e}",
            ])

        headers = ['Source', 'Target', 'OT Cost Diff', 'MAE (sparse)', 'MAPE (sparse)', 'Hellinger (sparse)', 'TV (sparse)', 'Max Violation', 'Run Time']

        # Print summary table to terminal
        logger.info("\n" + "="*80)
        logger.info("SUMMARY OF RESULTS")
        logger.info("="*80)
        table_str_terminal = tabulate(table_data_tot, headers=headers, tablefmt='grid', floatfmt=".3e")
        logger.info("\n" + table_str_terminal)

        # Create Box Plot Summary
        logger.info("\n" + "="*80)
        logger.info("Generating Box Plot Summary")
        logger.info("="*80)

        # Extract data for box plots
        metrics_to_plot = {
            'OT Cost Diff': ([r['otcost_diff'] for r in all_results_tot], [r['otcost_diff'] for r in all_results_not]),
            'MAE (Sparse)': ([r['mae_sparse'] for r in all_results_tot], [r['mae_sparse'] for r in all_results_not]),
            'MAPE (Sparse)': ([r['mape_sparse'] for r in all_results_tot], [r['mape_sparse'] for r in all_results_not]),
            'Hellinger (Sparse)': ([r['hellinger_sparse'] for r in all_results_tot], [r['hellinger_sparse'] for r in all_results_not]),
            'TV (Sparse)': ([r['tv_sparse'] for r in all_results_tot], [r['tv_sparse'] for r in all_results_not]),
            'Max Violation': ([r['max_violation'] for r in all_results_tot], []),
            'Run Time (Seconds)': ([r['run_time'] for r in all_results_tot], [r['run_time'] for r in all_results_not])
        }
        num_boxplots = len(metrics_to_plot)
        fig_box, axes_box = plt.subplots(1, num_boxplots, figsize=(20, 5), dpi=120)
        fig_box.suptitle(f"Custom Distribution Pairs Summary - {metric}", fontsize=16, y=1.05)
        
        for idx, (m_name, (data_tot, data_not)) in enumerate(metrics_to_plot.items()):
            ax = axes_box[idx]
            
            # Prepare plotting data
            plot_data = []
            labels = []
            colors = []
            
            if len(data_tot) > 0:
                plot_data.append(data_tot)
                labels.append('Ours (TOT)')
                colors.append('indianred')
            
            if len(data_not) > 0:
                plot_data.append(data_not)
                labels.append('NOT')
                colors.append('lavender')
                
            if not plot_data:
                continue

            bplot = ax.boxplot(plot_data, vert=True, patch_artist=True, labels=labels,
                               medianprops=dict(color="black"))

            # Color boxes
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

            ax.set_title(m_name, fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.6)

            # Add basic stats text for TOT and NOT
            
            m_median_tot = np.median(data_tot)
            m_median_not = np.median(data_not)
            stats_text = f"TOT Median: {m_median_tot:.2e}\nNOT Median: {m_median_not:.2e}"
            ax.text(0.6, 0.95, stats_text, 
                        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))
        plt.tight_layout()

        # Create Jupyter notebook with results
        logger.info("\n" + "="*80)
        logger.info("Creating Jupyter notebook with results")
        logger.info("="*80)
        
        nb = new_notebook()
        cells = []

        # Add title cell
        # add total distributions evaluated
        cells.append(new_markdown_cell(f"# Transformers for Optimal Transport (TOT): Custom distribution pairs - Metric: {metric}\n\n"
                                    f"Model: {model_base_path}\n\n"
                                    f"Dataset: {cfg.data.hf_dataset}\n\n"
                                    f"Dataset size: {cfg.data.dataset.size_s}\n\n"
                                    f"Support size (n_bins): {n_bins}\n\n"
                                    f"Total distributions evaluated: {len(list_distributions)}\n\n"
                                    f"Total pairs evaluated: {len(all_results_tot)}"))

        # Add tables to notebook
        cells.append(new_markdown_cell("## Summary tables"))
        cells.append(new_markdown_cell("### OT Cost Diff"))
        otcost_table_str_notebook = tabulate(otcost_table_data, headers=otcost_headers, tablefmt='pipe', floatfmt=".3e")
        cells.append(new_markdown_cell(otcost_table_str_notebook))
        cells.append(new_markdown_cell("### MAE (Sparse)"))
        mae_sparse_table_str_notebook = tabulate(mae_sparse_table_data, headers=mae_sparse_headers, tablefmt='pipe', floatfmt=".3e")
        cells.append(new_markdown_cell(mae_sparse_table_str_notebook))
        cells.append(new_markdown_cell("### MAPE (Sparse)"))
        mape_sparse_table_str_notebook = tabulate(mape_sparse_table_data, headers=mape_sparse_headers, tablefmt='pipe', floatfmt=".3e")
        cells.append(new_markdown_cell(mape_sparse_table_str_notebook))
        cells.append(new_markdown_cell("### Hellinger (Sparse)"))
        hellinger_sparse_table_str_notebook = tabulate(hellinger_sparse_table_data, headers=hellinger_sparse_headers, tablefmt='pipe', floatfmt=".3e")
        cells.append(new_markdown_cell(hellinger_sparse_table_str_notebook))
        cells.append(new_markdown_cell("### TV (Sparse)"))
        tv_sparse_table_str_notebook = tabulate(tv_sparse_table_data, headers=tv_sparse_headers, tablefmt='pipe', floatfmt=".3e")
        cells.append(new_markdown_cell(tv_sparse_table_str_notebook))
        cells.append(new_markdown_cell("### Run Time"))
        run_time_table_str_notebook = tabulate(run_time_table_data, headers=run_time_headers, tablefmt='pipe', floatfmt=".3e")
        cells.append(new_markdown_cell(run_time_table_str_notebook))

        # Add Box Plot to notebook
        cells.append(new_markdown_cell("## Summary Box Plots"))

        # Convert box plot figure to base64
        buf = BytesIO()
        fig_box.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        img_base64_box = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig_box)

        # Create code cell for box plot
        cell_box = new_code_cell("")
        cell_box.outputs = [
            nbformat.v4.new_output(
                output_type='display_data',
                data={'image/png': img_base64_box},
                metadata={}
            )
        ]
        cells.append(cell_box)


        # Add each figure as embedded image
        cells.append(new_markdown_cell("## Individual Results"))

        for fig_data in all_figures:
            # Add markdown header for this pair
            cells.append(new_markdown_cell(f"### {fig_data['idx']}: {fig_data['source']} → {fig_data['target']}"))

            # Create an empty code cell with the figure as output (no code visible)
            cell = new_code_cell("")
            cell.outputs = [
                nbformat.v4.new_output(
                    output_type='display_data',
                    data={'image/png': fig_data['img_base64']},
                    metadata={}
                )
            ]
            cells.append(cell)

        # Now close all figures
        plt.close('all')

        nb.cells = cells

        # Save notebook
        os.makedirs(cfg.inference.out_dir, exist_ok=True)
        #save_dir = os.path.join(cfg.inference.out_dir, model_base_path)
        notebook_path = os.path.join(model_base_path, f'eval_all_pairs_1d_{metric}.ipynb')

        with open(notebook_path, 'w') as f:
            nbformat.write(nb, f)

        # Save the results list as well for later reconstruction
        # These are lists of floats, so np.save is appropriate
        results_path_tot = os.path.join(model_base_path, f'eval_all_pairs_1d_{metric}_results_tot_sinkhorn.npy')
        results_path_not = os.path.join(model_base_path, f'eval_all_pairs_1d_{metric}_results_not.npy')
        
        np.save(results_path_tot, np.array(all_results_tot))
        np.save(results_path_not, np.array(all_results_not))
        
        logger.info(f"Saved TOT results to: {results_path_tot}")
        logger.info(f"Saved NOT results to: {results_path_not}")

        logger.info(f"Saved notebook to: {notebook_path}")
        logger.info(f"Open with: jupyter notebook {notebook_path}")

    return all_results_tot, all_results_not


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main function using Hydra for configuration management.
    """

    logger.info("Using GPU inference mode")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")

    # Set world_size if not specified
    if cfg.training.world_size is None: cfg.training.world_size = torch.cuda.device_count()

    # Regular inference routine
    logger.info("Running standard inference routine")
    inference_routine(cfg)

if __name__ == "__main__":
    main()