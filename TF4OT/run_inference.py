from TF4OT import tokenizers
from TF4OT import model
from TF4OT import utils
from TF4OT.utils import my_dtype_np, my_dtype_torch
from datasets import load_dataset
import torch
import functools
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from scipy import sparse
from tqdm import tqdm
import numpy as np
import logging
import yaml
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from tabulate import tabulate
import base64
from io import BytesIO
import itertools

# Add Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# Setup logging
logger = logging.getLogger(__name__)

row_wise = True

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
    if name: model_name = f"{name}.pth"
    else: model_name = utils.get_model_name_from_cfg(cfg)

    # Set default directory if not provided
    os.makedirs(cfg.model.save_path, exist_ok=True)

    # Combine directory with auto-generated filename
    full_save_path = os.path.join(cfg.model.save_path, model_name)
    logger.info(f"Auto-generated model name: {model_name}")
    
    # Loading MODEL
    logger.info(f"Loading model from: {full_save_path}")
    pytorch_model, tokenizer = load_model_by_name(full_save_path)

    # Loading TEST dataset
    logger.info("Loading dataset: " + cfg.data.hf_dataset)
    dataset     = load_dataset(cfg.data.hf_dataset, cache_dir=cfg.data.hf_cache_dir)
    test_data   = dataset["test"]
    test_loader = DataLoader(test_data,
                            batch_size=cfg.inference.batch_size,
                            shuffle=False,
                            num_workers=cfg.data.loading.num_workers,
                            pin_memory=(device.type == "cuda"),
                            collate_fn=utils.collate_from_string
    )
    logger.info(f"test_size={len(test_data)}, test.batch_size={cfg.inference.batch_size}, "
                f"batches_per_epoch={len(test_loader)}")

    # Create Tokenizer
    logger.info("Creating Tokenizer")
    # TODO: add support for more tokenizers
    OT_SOURCE_DIST         = "source_dist"
    OT_TARGET_DIST         = "target_dist"
    OT_INPUT_TOTAL         = "input_total"
    OT_INPUT_COST_MATRIX   = "input_cost_matrix"
    OT_TARGET_PLAN_MATRIX_SPARSE  = "target_otplan_solve_sparse"
    OT_TARGET_PLAN_MATRIX_DENSE   = "target_otplan_solve_dense"

    # Run distribution pair tests if enabled
    if hasattr(cfg.inference, 'test_custom_distributions') and cfg.inference.test_custom_distributions:
        logger.info(f"\n{'='*80}")
        logger.info("Starting distribution pair tests...")
        logger.info(f"{'='*80}\n")
        test_1d_distribution_examples(cfg, pytorch_model=pytorch_model, tokenizer=tokenizer, model_name=model_name)

    # Limit test batches if not testing full dataset
    if hasattr(cfg.inference, 'test_all_testset') and cfg.inference.test_all_testset:
        test_batches  = test_loader
        total_batches = len(test_loader)
    else:
        test_batches  = itertools.islice(test_loader, 10)
        total_batches = min(10, len(test_loader))

    pytorch_model.eval()

    # Initialize lists to collect metrics and figures
    all_mse_dense = []
    all_mse_sparse = []
    all_otcost_diff = []
    all_figures = []
    all_results = []
    
    # Import matplotlib for visualizations
    import matplotlib.pyplot as plt
    
    n_bins = cfg.data.dataset.support_size
    cost_fixed  = cfg.data.dataset.cost_fixed

    with torch.no_grad():
        pbar = tqdm(test_batches, total=total_batches)
        for batch_idx, batch in enumerate(pbar):

            # Run model INFERENCE
            encoder_input_ab = batch[OT_INPUT_TOTAL].to(device)
            if cost_fixed:
                encoder_input_c = None
            else:
                encoder_input_c = batch[OT_INPUT_COST_MATRIX].flatten(start_dim=1).to(device)
            text_ids_pred = pytorch_model.generate(encoder_input_ab=encoder_input_ab,
                                                   encoder_input_c=encoder_input_c, 
                                                   temperature=0.1)
            text_pred = tokenizer.from_tokens(text_ids_pred)
            text_pred = [t[:t.find(']')+1] if (']' in t) else t for t in text_pred]

            text_pred_dense = [sparse_row_string_to_dense_string(t, cfg.data.dataset.support_size, mode='index', precision=cfg.tokenizer.DigitTokenizer.decimal_precision) for t in text_pred ]
            ot_pred_dense   = np.stack([np.fromstring(t[1:-1], sep=',') for t in text_pred_dense])
            ot_pred_dense_np = ot_pred_dense.copy()  # Keep numpy version for visualization
            ot_pred_dense   = torch.tensor(ot_pred_dense, dtype=my_dtype_torch)
            ot_pred_sparse  = sparse.coo_matrix(ot_pred_dense)

            ot_target_batch  = batch[OT_TARGET_PLAN_MATRIX_DENSE]
            ot_target_dense  = ot_target_batch
            ot_target_dense_np = ot_target_dense.cpu().numpy()  # Keep numpy version for visualization
            ot_target_sparse = sparse.coo_matrix(ot_target_dense)

            # Calculate MSE diff
            mse_diff_dense  = mean_squared_error(ot_target_dense.flatten(), ot_pred_dense.flatten())
            mse_diff_sparse = utils.sparse_mse(ot_target_sparse, ot_pred_sparse)

            # Calculate OT cost diff
            C = batch[OT_INPUT_COST_MATRIX][0]
            C = C.detach().clone().to(dtype=my_dtype_torch)
            C_np = C.cpu().numpy()  # Keep numpy version for visualization
            otcost_gt     = (ot_target_dense * C).sum()
            otcost_pred   = (ot_pred_dense   * C).sum()
            otcost_diff   = abs(otcost_pred - otcost_gt).item()

            # Collect metrics
            all_mse_dense.append(mse_diff_dense)
            all_mse_sparse.append(mse_diff_sparse)
            all_otcost_diff.append(otcost_diff)
            
            # Store results for table
            all_results.append({
                'batch_idx': batch_idx,
                #'MSE diff dense': mse_diff_dense,
                'MSE Diff (sparse)': mse_diff_sparse,
                'OT Cost Diff': otcost_diff,
                #'OT Cost GT': otcost_gt.item(),
               # 'OT Cost Pred': otcost_pred.item()
            })
            
            # Create visualization for first sample in batch
            source = batch[OT_SOURCE_DIST][0].cpu().numpy()
            target = batch[OT_TARGET_DIST][0].cpu().numpy()
            
            # Only create figures for first few batches to avoid memory issues
            if batch_idx < min(20, total_batches):
                fig, axarr = plt.subplots(1, 5, figsize=(18, 3.5), dpi=120)
                fig.suptitle(f"Batch {batch_idx}", fontsize=16, y=1.02)
                
                # Titles
                axarr[0].set_title(f"Source $\\mu$", fontsize=12)
                axarr[1].set_title(f"Target $\\nu$", fontsize=12)
                axarr[2].set_title("Cost Matrix", fontsize=12)
                axarr[3].set_title(r"$\hat{\pi}$ Model Prediction", fontsize=12)
                axarr[4].set_title(r"$\pi$ Ground Truth", fontsize=12)
                
                # Common vmin/vmax for OT plans
                vmin = 0
                vmax = max(np.max(ot_pred_dense_np[0]), np.max(ot_target_dense_np[0]))
                
                # --- Source (bar) ---
                bin_edges = np.arange(len(source) + 1)
                axarr[0].bar(
                    bin_edges[:-1], source,
                    width=1.0, color="tab:blue", alpha=0.7, edgecolor="black"
                )
                axarr[0].set_xlim(0, len(source))
                axarr[0].set_ylim(0, max(source.max(), target.max()) * 1.1)
                axarr[0].set_xlabel("Bin index")
                axarr[0].set_ylabel("Probability")
                
                # --- Target (bar) ---
                bin_edges = np.arange(len(target) + 1)
                axarr[1].bar(
                    bin_edges[:-1], target,
                    width=1.0, color="tab:blue", alpha=0.7, edgecolor="black"
                )
                axarr[1].set_xlim(0, len(target))
                axarr[1].set_ylim(0, max(source.max(), target.max()) * 1.1)
                axarr[1].set_xlabel("Bin index")
                axarr[1].set_ylabel("Probability")
                
                # --- Cost matrix ---
                im2 = axarr[2].imshow(C_np, cmap='Blues', origin='lower', aspect="auto")
                axarr[2].set_xlabel("Target bins")
                axarr[2].set_ylabel("Source bins")
                plt.colorbar(im2, ax=axarr[2], fraction=0.046, pad=0.04)
                
                # --- Model Prediction ---
                im3 = axarr[3].imshow(ot_pred_dense.cpu().numpy(), cmap='Reds', vmin=vmin, vmax=vmax, 
                                     origin='lower', aspect="auto")
                axarr[3].set_xlabel("Target bins")
                axarr[3].set_ylabel("Source bins")
                axarr[3].text(0.02, 0.98, rf'$\sum \hat{{\pi}} = {ot_pred_sparse.sum():.5f}$', 
                             transform=axarr[3].transAxes, fontsize=10,
                             verticalalignment='top', bbox=dict(boxstyle='round', 
                             facecolor='white', alpha=0.8))
                plt.colorbar(im3, ax=axarr[3], fraction=0.046, pad=0.04)
                
                # --- Ground truth ---
                im4 = axarr[4].imshow(ot_target_dense.cpu().numpy(), cmap='Blues', vmin=vmin, vmax=vmax, 
                                     origin='lower', aspect="auto")
                axarr[4].set_xlabel("Target bins")
                axarr[4].set_ylabel("Source bins")
                axarr[4].text(0.02, 0.98, rf'$\sum \pi = {ot_target_sparse.sum():.5f}$', 
                             transform=axarr[4].transAxes, fontsize=10,
                             verticalalignment='top', bbox=dict(boxstyle='round', 
                             facecolor='white', alpha=0.8))
                plt.colorbar(im4, ax=axarr[4], fraction=0.046, pad=0.04)
                
                # Add error metrics text
                fig.text(0.5, -0.02, f'MSE diff (sparse): {mse_diff_sparse:.3e}  |  ' + 
                        f'OT Cost Diff: {otcost_diff:.3e}',
                        ha='center', fontsize=11, bbox=dict(boxstyle='round', 
                        facecolor='lightyellow', alpha=0.8))
                
                plt.tight_layout()
                
                # Store figure for notebook
                all_figures.append({
                    'fig': fig,
                    'batch_idx': batch_idx
                })
            
            # Update progress bar with current metrics
            pbar.set_postfix({
                'mse_dense': f'{mse_diff_dense:.6e}',
                'mse_sparse': f'{mse_diff_sparse:.6e}',
                'otcost_diff': f'{otcost_diff:.6e}'
            })
    
    # Convert to numpy arrays for statistics
    all_mse_dense = np.array(all_mse_dense)
    all_mse_sparse = np.array(all_mse_sparse)
    all_otcost_diff = np.array(all_otcost_diff)
    
    # Print statistics
    logger.info("\n" + "="*80)
    logger.info("INFERENCE METRICS SUMMARY")
    logger.info("="*80)
    logger.info(f"Number of batches evaluated: {len(all_mse_dense)}")
    logger.info("")
    
    # logger.info("MSE Diff (dense):")
    # logger.info(f"  Mean:   {all_mse_dense.mean():.6f}")
    # logger.info(f"  Median: {np.median(all_mse_dense):.6f}")
    # logger.info(f"  Min:    {all_mse_dense.min():.6f}")
    # logger.info(f"  Max:    {all_mse_dense.max():.6f}")
    # logger.info(f"  Std:    {all_mse_dense.std():.6f}")
    # logger.info("")
    
    logger.info("MSE Diff (sparse):")
    logger.info(f"  Mean:   {all_mse_sparse.mean():.6e}")
    logger.info(f"  Median: {np.median(all_mse_sparse):.6e}")
    logger.info(f"  Min:    {all_mse_sparse.min():.6e}")
    logger.info(f"  Max:    {all_mse_sparse.max():.6e}")
    logger.info(f"  Std:    {all_mse_sparse.std():.6e}")
    logger.info("")
    
    logger.info("OT Cost Difference:")
    logger.info(f"  Mean:   {all_otcost_diff.mean():.6e}")
    logger.info(f"  Median: {np.median(all_otcost_diff):.6e}")
    logger.info(f"  Min:    {all_otcost_diff.min():.6e}")
    logger.info(f"  Max:    {all_otcost_diff.max():.6e}")
    logger.info(f"  Std:    {all_otcost_diff.std():.6e}")
    logger.info("="*80 + "\n")
    
    # Prepare summary table data
    table_data = []
    for r in all_results:
        table_data.append([
            r['batch_idx'],
            f"{r['OT Cost Diff']:.3e}",
            #f"{r['MSE diff dense']:.3e}",
            f"{r['MSE Diff (sparse)']:.3e}",
            #f"{r['OT Cost GT']:.3e}",
            #f"{r['OT Cost Pred']:.3e}"
        ])
    
    headers = ['Batch', 'OT Cost Diff', 'MSE Diff (sparse)']
    
    # Print summary table to terminal
    logger.info("\n" + "="*80)
    logger.info("DETAILED RESULTS TABLE")
    logger.info("="*80)
    table_str_terminal = tabulate(table_data, headers=headers, tablefmt='grid', floatfmt=".3e")
    logger.info("\n" + table_str_terminal)
    
    # Create Jupyter notebook with results
    logger.info("\n" + "="*80)
    logger.info("Creating Jupyter notebook with results")
    logger.info("="*80)
    
    nb = new_notebook()
    cells = []
    
    # Add title cell
    cells.append(new_markdown_cell(
        f"# Test Set Inference Results\n\n"
        f"**Model:** {model_name}\n\n"
        f"**Total batches evaluated:** {len(all_results)}\n\n"
        f"**Dataset:** {cfg.data.hf_dataset}\n\n"
        f"**Support size (n_bins):** {n_bins}"
    ))
    
    # Add statistics summary
    stats_md = f"""## Summary Statistics

### MSE Sparse
- **Mean:** {all_mse_sparse.mean():.6e}
- **Median:** {np.median(all_mse_sparse):.6e}
- **Min:** {all_mse_sparse.min():.6e}
- **Max:** {all_mse_sparse.max():.6e}
- **Std:** {all_mse_sparse.std():.6e}

### OT Cost Difference
- **Mean:** {all_otcost_diff.mean():.6e}
- **Median:** {np.median(all_otcost_diff):.6e}
- **Min:** {all_otcost_diff.min():.6e}
- **Max:** {all_otcost_diff.max():.6e}
- **Std:** {all_otcost_diff.std():.6e}
"""
    cells.append(new_markdown_cell(stats_md))
    
    # Add summary table
    cells.append(new_markdown_cell("## Detailed Results Table"))
    table_str_notebook = tabulate(table_data, headers=headers, tablefmt='pipe', floatfmt=".3e")
    cells.append(new_markdown_cell(table_str_notebook))
    
    # Add individual visualizations
    cells.append(new_markdown_cell("## Individual Batch Visualizations"))
    
    for fig_data in all_figures:
        # Add markdown header for this batch
        cells.append(new_markdown_cell(f"### Batch {fig_data['batch_idx']}"))
        
        # Convert figure to base64 for embedding
        buf = BytesIO()
        fig_data['fig'].savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        
        plt.close(fig_data['fig'])
        
        # Create an empty code cell with the figure as output (no code visible)
        cell = new_code_cell("")
        cell.outputs = [
            nbformat.v4.new_output(
                output_type='display_data',
                data={'image/png': img_base64},
                metadata={}
            )
        ]
        cells.append(cell)
    
    # Close all remaining figures
    plt.close('all')
    
    nb.cells = cells
    
    # Save notebook
    out_dir = getattr(cfg.inference, 'out_dir', './inference_results')
    os.makedirs(out_dir, exist_ok=True)
    notebook_path = os.path.join(out_dir, f'inference_testset_{model_name}.ipynb')
    
    with open(notebook_path, 'w') as f:
        nbformat.write(nb, f)
    
    logger.info(f"Saved notebook to: {notebook_path}")
    logger.info(f"Open with: jupyter notebook {notebook_path}")
    logger.info("="*80 + "\n")


def load_model_by_name(full_save_path: str, device: str = "cuda"):
    """
    Load a TokenRegressor saved in:
      artifacts/{name}/model_weights.pth
      artifacts/{name}/config.json or config.yaml

    Returns: (model, tokenizer)
    """

    model_full_path  = full_save_path + ".pth"
    yaml_config_path = full_save_path + ".yaml"
    
    if os.path.exists(yaml_config_path):
        cfg = OmegaConf.load(yaml_config_path)
        cfg = OmegaConf.to_container(cfg, resolve=True)
    else:
        raise FileNotFoundError(f"No config file found at {yaml_config_path}")

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
    input_dim  = cfg["model"]["encoder"]["encoder_ab"]["input_dim"]
    emb_dim_ab = cfg["model"]["encoder"]["encoder_ab"]["emb_dim_ab"]
    encoder_ab = nn.Sequential(
        nn.Linear(input_dim, emb_dim_ab),
        nn.GELU(),
        nn.Linear(emb_dim_ab, emb_dim_ab)
    ).to(device)

    # Check if encoder_C exists in the config
    
    if cfg["data"]["dataset"]["cost_fixed"]:
        encoder_c = None
        emb_dim_c = 0
    else:
        input_dim = cfg["model"]["encoder"]["encoder_c"]["input_dim"]
        emb_dim_c = cfg["model"]["encoder"]["encoder_c"]["emb_dim_c"]
        encoder_c = nn.Sequential(
            nn.Linear(input_dim, emb_dim_c),
            nn.GELU(),
            nn.Linear(emb_dim_c, emb_dim_c)
        ).to(device)

    # Rebuild model using decoder config
    decoder_cfg     = cfg["model"]["decoder"]
    pytorch_model   = model.TokenRegressor(
        tokenizer   = tokenizer,
        encoder_ab  = encoder_ab,
        encoder_c   = encoder_c,
        emb_dim_ab  = emb_dim_ab,
        emb_dim_c   = emb_dim_c,
        emb_dim_d   = decoder_cfg["emb_dim_d"],
        max_seq_len = decoder_cfg["max_seq_len"],
        num_transformer_layers = decoder_cfg["num_transformer_layers"],
        num_attention_heads    = decoder_cfg["num_attention_heads"],
        dropout     = decoder_cfg.get("dropout", 0.0)  # Use default if not present
    ).to(device)

    # Load weights
    model_state = torch.load(model_full_path, map_location=device, weights_only=True)
    pytorch_model.load_state_dict(model_state, strict=True)

    return pytorch_model, tokenizer

def hist_from_samples(X_sampler, Y_sampler, bins=80, density=False, trim_percentiles=(0.5, 99.5)):
    """
    Returns:
        counts_X: (bins,) histogram for X
        counts_Y: (bins,) histogram for Y
        bin_edges: (bins+1,) shared edges
    """
    sample_size = 100_000
    # Sample
    X = X_sampler.sample(sample_size)
    Y = Y_sampler.sample(sample_size)

    # To flat numpy
    def to_1d_np(t):
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        return np.asarray(t).reshape(-1)

    X_np = to_1d_np(X)
    Y_np = to_1d_np(Y)

    # Shared bin edges (optionally trimmed for stability)
    if trim_percentiles is not None:
        p_lo, p_hi = trim_percentiles
        lo = np.percentile(np.concatenate([X_np, Y_np]), p_lo)
        hi = np.percentile(np.concatenate([X_np, Y_np]), p_hi)
        rng = (lo, hi)
    else:
        lo = np.min([X_np.min(), Y_np.min()])
        hi = np.max([X_np.max(), Y_np.max()])
        rng = (lo, hi)

    bin_edges = np.linspace(rng[0], rng[1], bins + 1)

    # Histograms with shared edges
    counts_X, _ = np.histogram(X_np, bins=bin_edges, density=density)
    probsX = counts_X / counts_X.sum()     # sums to 1 exactly (ignores out-of-range)
    counts_Y, _ = np.histogram(Y_np, bins=bin_edges, density=density)
    probsY = counts_Y / counts_Y.sum()     # sums to 1 exactly (ignores out-of-range)
    return probsX, probsY, bin_edges

def sparse_row_string_to_dense_string(
    s: str,
    n_bins: int,
    mode: str = "auto",           # 'auto' | 'index' | 'normalized'
    precision: int = 3,
    separator: str = ",",
    floatmode: str = "fixed",
    suppress_small: bool = True,
) -> str:
    """
    Convert a sparse row string to a dense row string "[..., ...]".

    Supported sparse formats:
      - 'index'      : [i0,j0,v0,i1,j1,v1,...]
      - 'normalized' : [x0,y0,v0,x1,y1,v1,...] with x=i/denom, y=j/denom,
                       where denom is auto-chosen as n_bins or n_bins-1.

    Duplicates for the same column are summed. Output has no spaces.
    """
    s = s.strip()
    #print("s: ", s)
    if s == "[]":
        #print("String is empty, returning zeros")
        dense = np.zeros(n_bins, dtype=float)
        return np.array2string(
            dense, precision=precision, separator=separator,
            floatmode=floatmode, suppress_small=suppress_small,
            max_line_width=1_000_000
        )

    if not s.startswith('['):
        # String doesn't start with '[', try to recover by adding the missing bracket
        #print(f"String doesn't start with '[', attempting recovery: {s}")
        
        # Try different recovery strategies
        recovery_attempts = []
        
        # Strategy 1: Add missing opening bracket if string ends with ']'
        if s.endswith(']'):
            recovery_attempts.append('[' + s)
        
        # Strategy 2: If string looks like it has numbers and commas, wrap it in brackets
        if any(c.isdigit() for c in s) and ',' in s:
            recovery_attempts.append('[' + s + ']' if not s.endswith(']') else '[' + s)
        
        # Strategy 3: If string is just a closing bracket or empty, treat as empty array
        if s.strip() in ['', ']']:
            recovery_attempts.append('[]')

        # Try each recovery attempt
        for attempt in recovery_attempts:
            try:
                # Test if this recovery attempt would work by trying to parse it
                test_inner = attempt[1:-1] if attempt.startswith('[') and attempt.endswith(']') else ""
                if test_inner:
                    test_vals = np.fromstring(test_inner, sep=',', dtype=float)
                    test_vals = test_vals[np.isfinite(test_vals)]
                    if test_vals.size > 0 and test_vals.size % 2 == 0:
                        #print(f"Recovery successful with attempt: {attempt}")
                        s = attempt
                        break
            except:
                continue
        else:
            # If all recovery attempts failed, return zeros
            #print("All recovery attempts failed, returning zeros")
            dense = np.zeros(n_bins, dtype=float)
            return np.array2string(
                dense, precision=precision, separator=separator,
                floatmode=floatmode, suppress_small=suppress_small,
                max_line_width=1_000_000
            )

    if not s.endswith(']'):
        # String starts with '[' but doesn't end with ']', try to recover parts
        # Find the last ']' or use the whole string if no ']' found
        #print("String doesn't end with ']', trying to recover parts")
        last_bracket = s.rfind(']')
        if last_bracket != -1:
            s = s[:last_bracket + 1]
        # If no ']' found, we'll try to parse what we have

    inner = s[1:-1]  # comma-separated, no spaces required
    #print("inner: ", inner)
    # Parse with robust error handling
    try:
        vals = np.fromstring(inner, sep=',', dtype=float)
        # Remove any NaN or infinite values that might have corrupted parsing
        vals = vals[np.isfinite(vals)]
    except:
        # If parsing fails completely, return zeros
        dense = np.zeros(n_bins, dtype=float)
        return np.array2string(
            dense, precision=precision, separator=separator,
            floatmode=floatmode, suppress_small=suppress_small,
            max_line_width=1_000_000
        )

    if vals.size == 0:
        # Empty after cleaning
        dense = np.zeros(n_bins, dtype=float)
        return np.array2string(
            dense, precision=precision, separator=separator,
            floatmode=floatmode, suppress_small=suppress_small,
            max_line_width=1_000_000
        )

    # Handle odd number of values gracefully instead of raising an error
    if vals.size % 2 != 0:
        # If odd, remove the last value to make it even (conservative approach)
        if vals.size > 1:
            vals = vals[:-1]
        else:
            # If only one value and odd, must be incomplete, so return zeros
            dense = np.zeros(n_bins, dtype=float)
            return np.array2string(
                dense, precision=precision, separator=separator,
                floatmode=floatmode, suppress_small=suppress_small,
                max_line_width=1_000_000
            )

    triples = vals.reshape(-1, 2)
    col_index = triples[:, 0]
    value = triples[:, 1]

    # Filter out invalid entries
    # Condition 1: Discard indices > n_bins
    valid_mask = col_index <= n_bins
    # Condition 2: Discard values > 1
    valid_mask = valid_mask & (value <= 1.0)

    # Apply the mask to filter valid entries
    col_index = col_index[valid_mask]
    value = value[valid_mask]

    #print(f"Filtered entries: {valid_mask.sum()} valid out of {len(valid_mask)} total")

    # Build dense row
    dense = np.zeros(n_bins, dtype=float)

    if mode == "index":
        # i,j as integer-like indices
        j_idx = np.rint(col_index).astype(int)
        cols = np.clip(j_idx, 0, n_bins - 1)
        np.add.at(dense, cols, value)

    # Format dense row as a compact string (no spaces)
    return np.array2string(
        dense, precision=precision, separator=separator,
        floatmode=floatmode, suppress_small=suppress_small,
        max_line_width=1_000_000
    )

def test_1d_distribution_examples(cfg: DictConfig, pytorch_model=None, tokenizer=None, model_name=None):

    """
    Test 1D distribution examples across multiple distribution pairs.

    Args:
        cfg: Hydra configuration
        pytorch_model: Pre-loaded model (optional). If None, will load from config.
        tokenizer: Pre-loaded tokenizer (optional). If None, will load from config.
    """
    import matplotlib.pyplot as plt
    import ot
    # --- tiny adapter to match hist_from_samplers API ---
    class NumpySampler:
        def __init__(self, fn, name=None):
            self.fn = fn  # fn(n) -> np.ndarray of shape (n,)
            self.name = name or "NumpySampler"
        def sample(self, n):
            x = self.fn(n).astype(np.float32)
            return torch.from_numpy(x).view(-1, 1)  # (n, 1)

    def zscore(x, eps=1e-8):
        x = (x - x.mean()) / (x.std() + eps)
        return x

    # --- distribution factories (all return NumpySampler with a .name) ---
    def BetaLeft(a=2.0, b=5.0):
        return NumpySampler(lambda n: zscore(np.random.beta(a, b, size=n)),
                            name=f"BetaLeft")

    def BetaRight(a=5.0, b=2.0):
        return NumpySampler(lambda n: zscore(np.random.beta(a, b, size=n)),
                            name=f"BetaRight")

    def LogNormal(mu=0.0, sigma=0.5):
        return NumpySampler(lambda n: zscore(np.random.lognormal(mean=mu, sigma=sigma, size=n)),
                            name=f"LogNormal")

    def Laplace(loc=0.0, scale=1.0):
        return NumpySampler(lambda n: zscore(np.random.laplace(loc, scale, size=n)),
                            name=f"Laplace")

    def StudentT(df=3):
        return NumpySampler(lambda n: zscore(np.random.standard_t(df, size=n)),
                            name=f"StudentT")

    def CauchyTrunc(x0=0.0, gamma=1.0, clip=6.0):
        def _f(n):
            x = np.random.standard_cauchy(size=n) * gamma + x0
            x = np.clip(x, -clip, clip)
            return zscore(x)
        return NumpySampler(_f, name=f"CauchyTrunc")

    def Triangular(left=-2.0, mode=0.0, right=2.0):
        return NumpySampler(lambda n: zscore(np.random.triangular(left, mode, right, size=n)),
                            name=f"Triangular")

    def UniformLowHigh(low=-2.0, high=2.0):
        return NumpySampler(lambda n: zscore(np.random.uniform(low, high, size=n)),
                            name=f"Uniform")

    def GaussianShifted(mu=1.5, sigma=1.0):
        return NumpySampler(lambda n: zscore(np.random.normal(mu, sigma, size=n)),
                            name=f"GaussianShifted")

    def GaussianMixture(means=(-2.0, 2.0), sigmas=(0.7, 0.7), weights=(0.5, 0.5)):
        means, sigmas, weights = np.array(means), np.array(sigmas), np.array(weights)
        weights = weights / weights.sum()
        def _f(n):
            comps = np.random.choice(len(weights), size=n, p=weights)
            x = np.random.normal(loc=means[comps], scale=sigmas[comps])
            return zscore(x)
        return NumpySampler(_f, name=f"GaussianMixture")

    def Exponential(scale=1.0):
        return NumpySampler(lambda n: zscore(np.random.exponential(scale, size=n)),
                            name=f"Exponential")

    def Weibull(a=1.5, scale=1.0):
        return NumpySampler(lambda n: zscore(scale * np.random.weibull(a, size=n)),
                            name=f"Weibull")

    def Gamma(shape=2.0, scale=1.0):
        return NumpySampler(lambda n: zscore(np.random.gamma(shape, scale, size=n)),
                            name=f"Gamma")

    def Pareto(a=2.5, clip=10.0):
        def _f(n):
            x = np.random.pareto(a, size=n)
            x = np.clip(x, 0, clip)
            return zscore(x)
        return NumpySampler(_f, name=f"Pareto")

    def Cosine():
        return NumpySampler(lambda n: np.cos(zscore(np.random.rand(n) * np.pi - np.pi/2)),
                            name="Cosine")

    def BimodalUniform(low1=-3, high1=-1, low2=1, high2=3, p=0.5):
        def _f(n):
            mask = np.random.rand(n) < p
            x = np.empty(n)
            x[mask] = np.random.uniform(low1, high1, size=mask.sum())
            x[~mask] = np.random.uniform(low2, high2, size=(~mask).sum())
            return zscore(x)
        return NumpySampler(_f, name=f"BimodalUniform")

    def GaussianMixtureWideNarrow():
        return GaussianMixture(means=(0, 0), sigmas=(0.3, 2.0), weights=(0.5, 0.5))

    # New sampler pairs to rotate through (source, target)
    sampler_pairs = [
        (BetaLeft(), BetaRight()),                     # skewed vs opposite skew
        (Laplace(), UniformLowHigh()),                 # heavy-tail vs uniform
        (StudentT(df=5), GaussianShifted(mu=1.0)),     # t vs shifted normal
        (LogNormal(sigma=0.6), Laplace(scale=1.5)),    # skewed heavy-tail vs laplace
        (CauchyTrunc(clip=5.0), StudentT(df=3)),       # very heavy-tail (trunc) vs heavy-tail
        (Triangular(-3, 0, 3), UniformLowHigh(-3, 3)), # triangular vs uniform
        (GaussianMixture(means=(-2, 2), sigmas=(0.7, 0.7), weights=(0.4, 0.6)),
        GaussianMixture(means=(-3, 0, 3), sigmas=(0.6, 0.5, 0.6), weights=(0.2, 0.6, 0.2))),
        # --- extra pairs ---
        (Exponential(scale=1.0), Weibull(a=2.0)),        # skew vs flexible skew
        (Gamma(2.0), LogNormal(mu=0.0, sigma=0.8)),      # gamma vs lognormal
        (Pareto(a=2.5), Laplace(scale=1.0)),             # fat tail vs double exponential
        (BimodalUniform(), GaussianMixture()),           # bimodal uniform vs Gaussian mixture
        (Cosine(), Triangular(-2, 0, 2)),                # bounded oscillatory vs triangular
        (GaussianMixtureWideNarrow(), StudentT(df=2)),   # wide/narrow mixture vs heavy tail
        (UniformLowHigh(-5, 5), Pareto(a=3.0)),          # flat vs power-law
    ]

    # Add reciprocal pairs immediately after each original pair
    sampler_pairs_with_reciprocals = []
    for a, b in sampler_pairs:
        sampler_pairs_with_reciprocals.append((a, b))
        sampler_pairs_with_reciprocals.append((b, a))

    # Load the pre-trained model if not provided
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect results across trials
    all_results = []
    all_figures = []  # Store figures for notebook
    n_samples   = 1_000_000
    n_bins      = cfg.data.dataset.support_size
    metric      = 'sqeuclidean'
    # check encoder C in config
    cost_fixed  = cfg.data.dataset.cost_fixed
    row_wise    = cfg.data.dataset.row_wise

    # Evaluate distribution pairs
    for idx, (X_sampler, Y_sampler) in enumerate(sampler_pairs_with_reciprocals):

        logger.info(f"Evaluating pair {idx+1}/{len(sampler_pairs_with_reciprocals)}: {X_sampler.name} -> {Y_sampler.name}")

        source, target, _ = hist_from_samples(
            X_sampler, Y_sampler, bins=n_bins, density=False
        )

        # Define support points
        if cost_fixed:
            # fixed C case
            xs = np.arange(n_bins, dtype=np.float32)[:, None]
            xt = np.arange(n_bins, dtype=np.float32)[:, None]
        else:
            x_low  = 0; x_high = 10
            xs = np.random.uniform(low=[x_low, x_low], high=[x_high, x_high], size=(n_bins, 2))
            xt = np.random.uniform(low=[x_low, x_low], high=[x_high, x_high], size=(n_bins, 2))

        # Define cost matrix
        C = ot.dist(xs, xt, metric=metric)
        C /= C.max()

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

                if not cost_fixed:
                    encoder_input_c.append(torch.tensor(C, dtype=my_dtype_torch).flatten(start_dim=0))

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

        encoder_input_ab    = torch.stack(encoder_input_ab).to(device)
        if cost_fixed:
            encoder_input_c = None
        else:
            encoder_input_c = torch.stack(encoder_input_c).to(device)

        # Run model INFERENCE
        text_ids_pred = pytorch_model.generate(
            encoder_input_ab=encoder_input_ab,
            encoder_input_c= encoder_input_c,
            temperature=0.1
        )
        text_pred = tokenizer.from_tokens(text_ids_pred)
        print( "idx: ", idx, "text_pred: ", text_pred)
        text_pred = [t[:t.find(']')+1] if (']' in t) else t for t in text_pred]

        text_pred_dense = [sparse_row_string_to_dense_string(t, n_bins, mode='index', precision=cfg.tokenizer.DigitTokenizer.decimal_precision) for t in text_pred ]
        ot_pred_dense = np.stack([np.fromstring(t[1:-1], sep=',') for t in text_pred_dense])
        # build sparse matrix directly
        ot_pred_sparse = sparse.coo_matrix(
            ot_pred_dense, shape=otplan_solve_sparse.shape
        )

        # Calculate MSE diff
        mse_diff_dense  = mean_squared_error(otplan_solve_dense.flatten(), ot_pred_dense.flatten())
        mse_diff_sparse = utils.sparse_mse(otplan_solve_sparse, ot_pred_sparse)
        # Calculate OT cost diff
        otcost_gt     = otplan_solve.value
        otcost_pred   = (ot_pred_dense * C).sum()
        otcost_diff   = abs(otcost_pred - otcost_gt)

        # Compute metrics
        ot_plan_pred = ot_pred_dense
        ot_plan_gt   = otplan_solve_dense

        all_results.append({
            'source': X_sampler.name,
            'target': Y_sampler.name,
            'OT Cost diff': otcost_diff,
            'MSE diff dense': mse_diff_dense,
            'MSE diff sparse': mse_diff_sparse
        })

        # Create visualization
        fig, axarr = plt.subplots(1, 5, figsize=(18, 3.5), dpi=120)
        fig.suptitle(f"{X_sampler.name} → {Y_sampler.name}", fontsize=16, y=1.02)

        # Titles
        axarr[0].set_title(f"Source $\\mu$ (1D)", fontsize=12)
        axarr[1].set_title(f"Target $\\nu$ (1D)", fontsize=12)
        axarr[2].set_title("Cost Matrix", fontsize=12)
        axarr[3].set_title(r"$\hat{\pi}$ Model Prediction", fontsize=12)
        axarr[4].set_title(r"$\pi$ Ground Truth (POT)", fontsize=12)

        # Common vmin/vmax for OT plans
        vmin = 0
        vmax = max(np.max(ot_plan_pred), np.max(ot_plan_gt))

        # --- Source (bar) ---
        bin_edges = np.arange(len(source) + 1)
        axarr[0].bar(
            bin_edges[:-1], source,
            width=1.0, color="tab:blue", alpha=0.7, edgecolor="black"
        )
        axarr[0].set_xlim(0, len(source))
        axarr[0].set_ylim(0, max(source.max(), target.max()) * 1.1)
        axarr[0].set_xlabel("Bin index")
        axarr[0].set_ylabel("Probability")

        # --- Target (bar) ---
        bin_edges = np.arange(len(target) + 1)
        axarr[1].bar(
            bin_edges[:-1], target,
            width=1.0, color="tab:blue", alpha=0.7, edgecolor="black"
        )
        axarr[1].set_xlim(0, len(target))
        axarr[1].set_ylim(0, max(source.max(), target.max()) * 1.1)
        axarr[1].set_xlabel("Bin index")
        axarr[1].set_ylabel("Probability")
        
        # --- Cost matrix ---
        im2 = axarr[2].imshow(C, cmap='Blues', origin='lower', aspect="auto")
        axarr[2].set_xlabel("Target bins")
        axarr[2].set_ylabel("Source bins")
        plt.colorbar(im2, ax=axarr[2], fraction=0.046, pad=0.04)

        # --- Model Prediction ---
        im3 = axarr[3].imshow(ot_plan_pred, cmap='Reds', vmin=vmin, vmax=vmax, 
                             origin='lower', aspect="auto")
        axarr[3].set_xlabel("Target bins")
        axarr[3].set_ylabel("Source bins")
        axarr[3].text(0.02, 0.98, rf'$\sum \hat{{\pi}} = {ot_pred_sparse.sum():.5f}$', 
                     transform=axarr[3].transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round', 
                     facecolor='white', alpha=0.8))
        plt.colorbar(im3, ax=axarr[3], fraction=0.046, pad=0.04)

        # --- Ground truth ---
        im4 = axarr[4].imshow(ot_plan_gt, cmap='Blues', vmin=vmin, vmax=vmax, 
                             origin='lower', aspect="auto")
        axarr[4].set_xlabel("Target bins")
        axarr[4].set_ylabel("Source bins")
        axarr[4].text(0.02, 0.98, rf'$\sum \pi = {otplan_solve_sparse.sum():.5f}$', 
                     transform=axarr[4].transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round', 
                     facecolor='white', alpha=0.8))
        plt.colorbar(im4, ax=axarr[4], fraction=0.046, pad=0.04)
        
        # Add error metrics text
        fig.text(0.5, -0.02, f'MSE Diff (sparse): {mse_diff_sparse:.3e}  |  ' + 
                f'OT Cost Diff: {otcost_diff:.3e}',
                ha='center', fontsize=11, bbox=dict(boxstyle='round', 
                facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        
        # Store figure for notebook
        all_figures.append({
            'fig': fig,
            'source': X_sampler.name,
            'target': Y_sampler.name,
            'idx': idx
        })

    # Prepare summary table data
    table_data = []
    for r in all_results:
        table_data.append([
            r['source'],
            r['target'],
            f"{r['OT Cost diff']:.3e}",
            f"{r['MSE diff sparse']:.3e}"
        ])

    headers = ['Source', 'Target', 'OT Cost Diff', 'MSE Diff (dense)', 'MSE Diff (sparse)']

    # Print summary table to terminal
    logger.info("\n" + "="*80)
    logger.info("SUMMARY OF RESULTS")
    logger.info("="*80)
    table_str_terminal = tabulate(table_data, headers=headers, tablefmt='grid', floatfmt=".3e")
    logger.info("\n" + table_str_terminal)
    
    # Create Jupyter notebook with results
    logger.info("\n" + "="*80)
    logger.info("Creating Jupyter notebook with results")
    logger.info("="*80)
    
    nb = new_notebook()
    cells = []
    
    # Add title cell
    cells.append(new_markdown_cell("# Optimal Transport Inference Results\n\n"
                                   f"Total pairs evaluated: {len(all_results)}"))
    
    # Add summary table
    cells.append(new_markdown_cell("## Summary Table"))
    table_str_notebook = tabulate(table_data, headers=headers, tablefmt='pipe', floatfmt=".3e")
    cells.append(new_markdown_cell(table_str_notebook))
    
    # Add each figure as embedded image
    cells.append(new_markdown_cell("## Individual Results"))
    
    for fig_data in all_figures:
        # Add markdown header for this pair
        cells.append(new_markdown_cell(f"### {fig_data['idx']}: {fig_data['source']} → {fig_data['target']}"))
        
        # Convert figure to base64 for embedding
        buf = BytesIO()
        fig_data['fig'].savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        plt.close(fig_data['fig'])
        
        # Create an empty code cell with the figure as output (no code visible)
        cell = new_code_cell("")
        cell.outputs = [
            nbformat.v4.new_output(
                output_type='display_data',
                data={'image/png': img_base64},
                metadata={}
            )
        ]
        cells.append(cell)
    
    # Now close all figures
    import matplotlib.pyplot as plt
    plt.close('all')

    nb.cells = cells

    # Save notebook
    os.makedirs(cfg.inference.out_dir, exist_ok=True)
    notebook_path = os.path.join(cfg.inference.out_dir, f'inference_custom_{model_name}.ipynb')

    with open(notebook_path, 'w') as f:
        nbformat.write(nb, f)

    logger.info(f"Saved notebook to: {notebook_path}")
    logger.info(f"Open with: jupyter notebook {notebook_path}")

    return all_results

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