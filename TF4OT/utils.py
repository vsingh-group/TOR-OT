import torch
import numpy as np
from scipy import sparse
#mydtype = torch.float32

my_dtype_np    = np.float32
my_dtype_torch = torch.float32

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

def sparse_mse(A: sparse.coo_matrix, B: sparse.coo_matrix) -> float:
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
    diff = A - B
    sq_norm = diff.multiply(diff).sum()
    
    # Count non-zero elements in union of A and B
    union = A.astype(bool) + B.astype(bool)
    nnz = union.nnz
    
    if nnz == 0:
        return 0.0
    
    return float(sq_norm) / nnz

def get_model_name_from_cfg(cfg):
    return (
            f"model_data{cfg.data.dataset.size_s}"
            f"_prec{cfg.tokenizer.DigitTokenizer.decimal_precision}"
            f"_support{cfg.data.dataset.support_size}"
            f"_ep{cfg.training.num_epochs}"
            f"_dec_units_{cfg.model.decoder.units}"
            f"_nlayers_{cfg.model.decoder.num_transformer_layers}"
            f"_nheads_{cfg.model.decoder.num_attention_heads}"
            f"_drop_{cfg.model.decoder.dropout}"
        )    