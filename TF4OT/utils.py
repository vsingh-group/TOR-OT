import torch
import numpy as np
from scipy import sparse
import sklearn.metrics
import logging
import os
import sys
import io

#mydtype = torch.float32
my_dtype_np    = np.float32
my_dtype_torch = torch.float32
metrics = ['euclidean', 'sqeuclidean', 'cityblock', 'chebyshev', 'canberra', 'minkowski']


class RankFilter(logging.Filter):
    def __init__(self, rank:int): self.rank = rank
    def filter(self, record):      record.rank = self.rank; return True

def setup_logging(log_path: str | None, rank: int):
    """Console logs on all ranks; file logs only on rank 0."""
    root = logging.getLogger()
    root.handlers.clear()
#    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [rank %(rank)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    rf = RankFilter(rank)

    # Console for everyone
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO); ch.setFormatter(fmt); ch.addFilter(rf)
    root.addHandler(ch)

    # File only on rank 0
    if rank == 0 and log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO); fh.setFormatter(fmt); fh.addFilter(rf)
        root.addHandler(fh)

# --- redirect tqdm to logging ---
class TqdmToLogger(io.TextIOBase):
    """Redirect tqdm writes to a logger (one line per refresh)."""
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
    def write(self, buf):
        buf = buf.strip().replace("\r", "")
        if buf:
            self.logger.log(self.level, buf)
    def flush(self):  # no-op to satisfy file-like API
        pass        

def collate_from_double(batch):
    """
    Custom collate_fn that stacks fixed-size tensors but leaves
    variable-length lists (like 'output_map_sparse') as Python lists.
    """
    elem = batch[0]
    batch_out = {}
    for key in elem:
        values = [d[key] for d in batch]
        # Variable-length lists (e.g., sparse rows)
        if isinstance(values[0], list) and len(set(len(v) for v in values)) > 1:
            batch_out[key] = values  # keep as list
        # Fixed-length lists
        else:
            try:
                #batch_out[key] = default_collate(values)
                batch_out[key] = torch.tensor(values, dtype=my_dtype_torch)
            except Exception:
                # fallback for non-stackable items
                batch_out[key] = torch.tensor(values, dtype=my_dtype_torch)
    return batch_out

def collate_from_string(batch):
    """
    Custom collate_fn that stacks fixed-size tensors but leaves
    variable-length lists (like 'output_map_sparse') as Python lists.
    """
    elem = batch[0]
    batch_out = {}
    for key in elem:
        values = [d[key] for d in batch]
        # Variable-length lists (e.g., sparse rows)
        if isinstance(values[0], list) and len(set(len(v) for v in values)) > 1:
            batch_out[key] = values  # keep as list
        # Fixed-length lists
        else:
            try:
                #batch_out[key] = default_collate(values)
                batch_out[key] = torch.tensor(values)
            except Exception:
                # fallback for non-stackable items
                batch_out[key] = torch.tensor(values, dtype=my_dtype_torch)
    return batch_out

def to_string_lists(data, decimals=3):
    """
    Convert a list of lists into formatted string lists.
    Assumes each inner list alternates between int and float values.
    
    Args:
        data (list[list[float]]): The input list of lists.
        decimals (int): Number of decimal digits to keep for floats.

    Returns:
        list[str]: List of formatted strings, e.g. ["[0, 0.0487, 1, 0.0607]", ...]
    """
    formatted_list = []
    float_fmt = f"{{:.{decimals}f}}"  # float format, e.g., "{:.3f}"

    for sublist in data:
        formatted_sub = []
        for i, x in enumerate(sublist):
            # If even index (0, 2, 4...), format as integer
            if i % 2 == 0:
                formatted_sub.append(str(int(x)))
            # If odd index (1, 3, 5...), format as float
            else:
                formatted_sub.append(float_fmt.format(x))
        # Join with commas and brackets
        formatted_str = "[" + ", ".join(formatted_sub) + "]"
        formatted_list.append(formatted_str)

    return formatted_list

def parse_predicted_sparse_row_string(
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


def hist_from_samples(X_sample, Y_sample, bins=80, density=False, trim_percentiles=(0.5, 99.5)):
    """
    Returns:
        counts_X: (bins,) histogram for X
        counts_Y: (bins,) histogram for Y
        bin_edges: (bins+1,) shared edges
    """


    # To flat numpy
    def to_1d_np(t):
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        return np.asarray(t).reshape(-1)

    X_np = to_1d_np(X_sample)
    Y_np = to_1d_np(Y_sample)

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
    return probsX, probsY, rng

# MAE
def sparse_mean_abs_error(A: sparse.coo_matrix, B: sparse.coo_matrix) -> float:
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
    diff = (A - B).tocoo()
    mae  = np.abs(diff.data).mean(dtype=np.float64)
    return mae
# MAPE
def sparse_mean_abs_percentage_error(
    A: sparse.coo_matrix, #target
    B: sparse.coo_matrix, #prediction
    eps: float = 1e-12,
) -> float:
    """
    MAPE over the union of nonzero locations in A and B.
    Missing entries are treated as 0.

    Computes mean(|A - B| / max(|A|, eps)) over (i,j) where A_ij != 0 or B_ij != 0.
    """
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

    # Canonicalize COO (merge duplicates)
    A = A.tocoo().copy()
    B = B.tocoo().copy()

    # Union support via sparse difference (keeps only places where they differ),
    # plus we still need places where A==B!=0. So we explicitly union indices.
    # Build union index set in a vectorized way.
    idxA = A.row.astype(np.int64) * A.shape[1] + A.col.astype(np.int64)
    idxB = B.row.astype(np.int64) * B.shape[1] + B.col.astype(np.int64)
    idxU = np.union1d(idxA, idxB)

    if idxU.size == 0:
        return 0.0

    # Convert both to CSR for fast indexed lookup
    A_csr = A.tocsr()
    B_csr = B.tocsr()

    rows = (idxU // A.shape[1]).astype(np.int32)
    cols = (idxU %  A.shape[1]).astype(np.int32)

    # Gather values at union locations (sparse indexing returns a matrix)
    y_true = np.asarray(A_csr[rows, cols]).ravel().astype(np.float64)
    y_pred = np.asarray(B_csr[rows, cols]).ravel().astype(np.float64)

    # Make MAPE well-defined when y_true == 0:
    # sklearn already uses an internal epsilon, but we can be explicit and stable.
    # This implements |y_true - y_pred| / max(|y_true|, eps) through sklearn by
    # scaling y_true away from exact zeros.
    # (Equivalent to clipping denominator)
    y_true_safe = np.where(np.abs(y_true) < eps, np.sign(y_true) * eps, y_true)
    y_true_safe = np.where(y_true_safe == 0.0, eps, y_true_safe)

    mape_score = sklearn.metrics.mean_absolute_percentage_error(y_true_safe, y_pred)
    mape_score = float(mape_score)

    return mape_score

# wMAPE
def sparse_weighted_mean_abs_percentage_error(
    A: sparse.coo_matrix, #target
    B: sparse.coo_matrix, #prediction
    eps: float = 1e-12,
) -> float:
    """
    MAPE over the union of nonzero locations in A and B.
    Missing entries are treated as 0.

    Computes mean(|A - B| / max(|A|, eps)) over (i,j) where A_ij != 0 or B_ij != 0.
    """
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

    # Canonicalize COO (merge duplicates)
    A = A.tocoo().copy()
    B = B.tocoo().copy()

    # Union support via sparse difference (keeps only places where they differ),
    # plus we still need places where A==B!=0. So we explicitly union indices.
    # Build union index set in a vectorized way.
    idxA = A.row.astype(np.int64) * A.shape[1] + A.col.astype(np.int64)
    idxB = B.row.astype(np.int64) * B.shape[1] + B.col.astype(np.int64)
    idxU = np.union1d(idxA, idxB)

    if idxU.size == 0:
        return 0.0

    # Convert both to CSR for fast indexed lookup
    A_csr = A.tocsr()
    B_csr = B.tocsr()

    rows = (idxU // A.shape[1]).astype(np.int32)
    cols = (idxU %  A.shape[1]).astype(np.int32)

    # Gather values at union locations (sparse indexing returns a matrix)
    y_true = np.asarray(A_csr[rows, cols]).ravel().astype(np.float64)
    y_pred = np.asarray(B_csr[rows, cols]).ravel().astype(np.float64)

    # Make MAPE well-defined when y_true == 0:
    # sklearn already uses an internal epsilon, but we can be explicit and stable.
    # This implements |y_true - y_pred| / max(|y_true|, eps) through sklearn by
    # scaling y_true away from exact zeros.
    # (Equivalent to clipping denominator)
    y_true_safe = np.where(np.abs(y_true) < eps, np.sign(y_true) * eps, y_true)
    y_true_safe = np.where(y_true_safe == 0.0, eps, y_true_safe)


    # Calculate the sum of absolute errors
    absolute_errors_sum = np.sum(np.abs(y_true_safe - y_pred))
    
    # Calculate the sum of actual values
    actual_values_sum = np.sum(np.abs(y_true_safe))
    
    # Avoid division by zero
    if actual_values_sum == 0:
        return 0.0  # Or handle as appropriate for your use case
        
    wmape = absolute_errors_sum / actual_values_sum

    return float(wmape)

# Hellinger
def sparse_hellinger_distance(P: sparse.coo_matrix, Q: sparse.coo_matrix) -> float:
    """
    Hellinger distance between two discrete probability distributions stored as sparse COO matrices.

    Assumes P, Q are nonnegative and (approximately) sum to 1 (not strictly required).
    Returns a float in [0, 1] (up to numerical tolerance) if P and Q are valid distributions.
    """
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: {P.shape} vs {Q.shape}")

    P = P.tocoo()
    Q = Q.tocoo()

    # sqrt on nonzeros (zeros stay zero)
    sqrtP = sparse.coo_matrix((np.sqrt(np.clip(P.data, 0.0, None)), (P.row, P.col)), shape=P.shape)
    sqrtQ = sparse.coo_matrix((np.sqrt(np.clip(Q.data, 0.0, None)), (Q.row, Q.col)), shape=Q.shape)

    diff = (sqrtP - sqrtQ).tocoo()

    # ||diff||_2 / sqrt(2)
    sq_sum = np.dot(diff.data, diff.data).astype(np.float64)  # sum of squares of nonzeros
    return float(np.sqrt(sq_sum) / np.sqrt(2.0))

# Total Variation
def sparse_total_variation(P: sparse.coo_matrix, Q: sparse.coo_matrix) -> float:
    """
    Total Variation distance between two sparse 2D discrete distributions:
        TV(P,Q) = 0.5 * sum_{i,j} |P_ij - Q_ij|

    Works without densifying; sums over the full grid implicitly via the sparse difference.
    """
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: {P.shape} vs {Q.shape}")

    P = P.tocoo().copy()
    Q = Q.tocoo().copy()

    diff = (P - Q).tocoo()
    return float(0.5 * np.abs(diff.data).sum(dtype=np.float64))

def compute_constraint_violations(pred_plan_sparse, source_marg, target_marg):
    """
    Compute constraint violations for an OT plan.
    
    Args:
        pred_plan_sparse: scipy.sparse.coo_matrix (N, N)
        source_marg: np.ndarray (N,) or torch.Tensor
        target_marg: np.ndarray (N,) or torch.Tensor
        
    Returns:
        dict with keys: 'row_violation', 'col_violation', 'max_violation'
    """
    # Ensure marginals are numpy arrays (flattened)
    if isinstance(source_marg, torch.Tensor):
        source_marg = source_marg.detach().cpu().numpy().flatten()
    else:
        source_marg = np.asarray(source_marg).flatten()
        
    if isinstance(target_marg, torch.Tensor):
        target_marg = target_marg.detach().cpu().numpy().flatten()
    else:
        target_marg = np.asarray(target_marg).flatten()
    
    # Calculate row and column sums of the predicted plan
    # sum(axis=1) returns matrix of shape (N, 1), need flattening
    row_sums = np.asarray(pred_plan_sparse.sum(axis=1)).flatten()
    col_sums = np.asarray(pred_plan_sparse.sum(axis=0)).flatten()
    
    # Compute L1 norm of errors
    # Error = || P 1 - a ||_1
    row_err = np.abs(row_sums - source_marg).sum()
    col_err = np.abs(col_sums - target_marg).sum()
    
    # Max violation (max absolute error in any row or column sum? 
    # Or max of the total row/col errors? usually max of the L1 errors or L_inf error?
    # User said: "max violation (constraint violation in colums and rows)" 
    # Usually this implies L_infinity norm of the marginal difference vectors, 
    # but the prompt says "max(row_err, col_err)" in my thought process? 
    # Wait, the prompt said: "Report || pi 1 - a ||_1, || pi^T 1 - b ||_1, max violation"
    # "max violation" is typically max(|row_sums - source| + |col_sums - target|)? 
    # Or max( ||..||_inf )? 
    # Let's compute the L_inf violation as well, which is often what "max violation" refers to in solvers (primal feasibility).
    # But for a single scalar "max_violation", let's assume it means the maximum absolute deviation in any single marginal constraint.
    
    max_row_dev = np.abs(row_sums - source_marg).max()
    max_col_dev = np.abs(col_sums - target_marg).max()
    max_violation = max(max_row_dev, max_col_dev)
    
    return {
        'row_violation': float(row_err),
        'col_violation': float(col_err),
        'max_violation': float(max_violation)
    }

def print_sparse_matrix(A):
    A = A.tocoo()
    for r, c, v in zip(A.row, A.col, A.data):
        print(f"({r}, {c}), {v}")

def print_sparse_matrices(A, B, C, precision_to_display=4):
    # Convert to COO to easily extract (row, col) pairs
    A_coo = A.tocoo()
    B_coo = B.tocoo()
    C_coo = C.tocoo()

    # Collect set of all unique indices (r, c) present in any matrix
    indices = set(zip(A_coo.row, A_coo.col))
    indices.update(zip(B_coo.row, B_coo.col))
    indices.update(zip(C_coo.row, C_coo.col))
    
    # Sort for deterministic output order
    sorted_indices = sorted(list(indices))

    # Convert to CSR for efficient point lookup
    A_csr = A.tocsr()
    B_csr = B.tocsr()
    C_csr = C.tocsr()

    rows = ["COORDS,\t\tTOT,\tNOT,\tPOT_GT"]
    for r, c in sorted_indices:
        v_A = A_csr[r, c]
        v_B = B_csr[r, c]
        v_C = C_csr[r, c]
        #print(f"({r}, {c}),\t\t{v_A:.{precision_to_display}f},\t\t{v_B:.{precision_to_display}f},\t\t{v_C:.{precision_to_display}f}")
        rows.append(f"({r:02d}, {c:02d}),\t{v_A:.{precision_to_display}f},\t{v_B:.{precision_to_display}f},\t{v_C:.{precision_to_display}f}")
    
    if rows:
        logger = logging.getLogger(__name__)
        logger.info("\n" + "\n".join(rows))


def get_subfolder_name_from_cfg(cfg):
    return (
            f"model_dim{cfg.model.decoder.emb_dim_d}"
            f"_context{cfg.model.decoder.max_seq_len}"
            f"_tflayers{cfg.model.decoder.num_transformer_layers}"
            f"_tfheads{cfg.model.decoder.num_attention_heads}"
            f"_num_prefix{cfg.model.num_prefix_tokens}"
            f"_dropout{cfg.model.decoder.dropout}"
            f"_flash{cfg.model.use_flash_attention if cfg.model.use_flash_attention else 'False'}"
            f"_data{cfg.data.dataset.size_s}"
            f"{'_rowwise' if cfg.data.dataset.row_wise else ''}"
            f"_cost{cfg.data.dataset.input_cost_type}"
            f"_prec{cfg.tokenizer.DigitTokenizer.decimal_precision}"
            f"_bins{cfg.data.dataset.support_size}"
            f"_{'DigitTokenizer' if hasattr(cfg.tokenizer, 'DigitTokenizer') else 'IEEEFloatTokenizer'}"
            f"_epochs{cfg.training.num_epochs}"
            f"_batch{cfg.training.batch_size_per_device}"
            f"{'_' + cfg.data.dataset.input_cost_metric if cfg.data.dataset.input_cost_type in ['fixed', 'variable'] else ''}"
    )

class HistogramSampler:
    """
    Sampler that draws samples from a 1D histogram.
    Used to interface discrete dataset distributions with Neural OT code which expects continuous samplers.
    """
    def __init__(self, histogram, ranges=(0, 1), device='cuda'):
        self.histogram = histogram / histogram.sum()
        self.ranges = ranges
        self.device = device
        self.n_bins = len(histogram)
        self.bin_edges = torch.linspace(ranges[0], ranges[1], self.n_bins + 1, device=device)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        
    def sample(self, batch_size):
        # Sample bins based on histogram probabilities
        bin_indices = torch.multinomial(self.histogram, batch_size, replacement=True)
        
        # Add uniform noise within the bin to make it continuous-ish
        # Or just return bin centers? NOT typically expects continuous samples.
        # Let's return uniform samples within the selected bins.
        
        bin_width = self.bin_edges[1] - self.bin_edges[0]
        offsets = torch.rand(batch_size, device=self.device) * bin_width
        
        lower_edges = self.bin_edges[bin_indices]
        samples = lower_edges + offsets
        
        return samples.view(-1, 1) # (B, 1)
