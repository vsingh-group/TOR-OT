import torch


def collate_from_double(batch):
    """
    Custom collate_fn that stacks fixed-size tensors but leaves
    variable-length lists (like 'output_map_sparse') as Python lists.
    """
    mydtype = torch.float32
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
                batch_out[key] = torch.tensor(values, dtype=mydtype)
            except Exception:
                # fallback for non-stackable items
                batch_out[key] = torch.tensor(values, dtype=mydtype)
    return batch_out

def collate_from_string(batch):
    """
    Custom collate_fn that stacks fixed-size tensors but leaves
    variable-length lists (like 'output_map_sparse') as Python lists.
    """
    mydtype = torch.float32
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
                batch_out[key] = torch.tensor(values, dtype=mydtype)
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