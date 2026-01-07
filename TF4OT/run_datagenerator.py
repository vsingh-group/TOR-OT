#!/usr/bin/env python3
"""
Script to create and save a Hugging Face dataset for Optimal Transport (OT) mapping.

This script generates synthetic data for learning optimal transport mappings between
probability distributions using sparse matrix representations.
"""

import ot
import json
import torch
import random
import argparse
import numpy as np
import scipy.sparse as sp
from TF4OT import utils
from pathlib import Path
from typing import Dict
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo
from TF4OT.utils import my_dtype_np, my_dtype_torch

def coo_tolist(A: sp.coo_matrix, j: int) -> list:
    """
    Extracts the j-th row of a COO sparse matrix as a compact list:
        [col1, val1, col2, val2, ...]
    containing only nonzero entries.

    Parameters
    ----------
    A : sp.coo_matrix
        Input sparse matrix in COO format.
    j : int
        Row index to extract.

    Returns
    -------
    list
        Flat list alternating column indices and values for row j.
    """
    # Ensure the matrix is in COO format
    if not sp.isspmatrix_coo(A):
        A = A.tocoo(copy=False)

    sparse_rep = []
    if j is None:
        for r, c, v in zip(A.row, A.col, A.data):
            sparse_rep.extend([r, c, v])
    else:
        # Select entries from the j-th row
        mask = (A.row == j)
        cols = A.col[mask]
        vals = A.data[mask]
        # Interleave column indices and values
        for c, v in zip(cols, vals): 
            sparse_rep.extend([c, v])

    return sparse_rep

def ot_sample_generator(n_bins: int, dataset_size: int, metric = 'sqeuclidean', cost_type='fixed', row_wise=True, device: str = 'cpu', dim: int = 1):
    """
    Generator function that yields individual samples for the optimal transport dataset.

    Args:
        n_bins: Number of bins for discretization
        dataset_size: Number of samples to generate
        device: Device to use for tensor operations

    Yields:
        Dictionary containing a single sample
    """

    print(f"Generating {dataset_size} samples with {n_bins} bins...")

    # Generate source distribution
    print("Generating source and target distributions...")
    source_dist = np.random.uniform(size=(dataset_size, n_bins))
    source_dist /= source_dist.sum(axis=1, keepdims=True)
    # Generate target distribution
    target_dist = np.random.uniform(size=(dataset_size, n_bins))
    target_dist /= target_dist.sum(axis=1, keepdims=True)

    # Define support points
    possible_metrics = utils.metrics
    metric_one_hot  = [1.0 if m == metric else 0.0 for m in possible_metrics]

    # Fixed cost matrix case - same C for all samples
    if cost_type == 'fixed':
        if dim == 1:
            xs = np.arange(n_bins, dtype=my_dtype_np)[:, None]
            xt = np.arange(n_bins, dtype=my_dtype_np)[:, None]
        elif dim == 2:
            # Replicate logic from run_inference_2d.py L1725-L1751 for consistency
            rng = [[-2.5, 2.5], [-2.5, 2.5]]
            s = int(np.sqrt(n_bins))
            if s*s != n_bins:
                print(f"Warning: n_bins {n_bins} is not a perfect square and not 30. Using {s}x{s}={s*s} grid points.")
            nx, ny = s, s
            
            dx = (rng[0][1] - rng[0][0]) / nx
            dy = (rng[1][1] - rng[1][0]) / ny
            xc = np.linspace(rng[0][0] + dx/2, rng[0][1] - dx/2, nx)
            yc = np.linspace(rng[1][0] + dy/2, rng[1][1] - dy/2, ny)
            
            # Grid points matching flatten order (indexing='ij')
            XX, YY = np.meshgrid(xc, yc, indexing='ij') 
            grid_points = np.stack([XX.flatten(), YY.flatten()], axis=1) # (n_bins, 2)
            
            # If n_bins != nx*ny (e.g. if we used s*s but n_bins was something else), resize?
            # Ideally we fail or warn. For now assuming consistent usage.
            if n_bins != nx * ny:
                print(f"Warning: n_bins {n_bins} != {nx}*{ny}. Grid size mismatch.")
                
            xs = grid_points
            xt = grid_points

        # Define cost matrix    
        C = ot.dist(xs, xt, metric=metric, use_tensor=True, p=3)
        C /= C.max()

    print("Computing optimal transport maps and yielding samples...")
    for i in tqdm(range(dataset_size), desc="Processing datasets"):
        # Variable cost matrix case
        if cost_type == 'variable':
            # n_bins defined elsewhere
            x_low = 0; x_high = n_bins
            # integers from [x_low, x_high) -> 0 to 9
            xs = np.random.randint(low=x_low, high=x_high, size=(n_bins, dim))
            xt = np.random.randint(low=x_low, high=x_high, size=(n_bins, dim))
            C = ot.dist(xs, xt, metric=metric, use_tensor=True, p=3)
            C = C / C.max() # avoid casting errors later
        elif cost_type == 'semivariable':
            xs = np.zeros((n_bins, dim))
            xt = np.zeros((n_bins, dim))
            
            if dim == 1:
                xs[:, 0] = np.arange(n_bins, dtype=my_dtype_np)
                xt[:, 0] = np.arange(n_bins, dtype=my_dtype_np)
            elif dim == 2:
                 s = int(np.sqrt(n_bins))
                 x = np.arange(s, dtype=my_dtype_np)
                 y = np.arange(s, dtype=my_dtype_np)
                 xv, yv = np.meshgrid(x, y)
                 grid = np.stack((xv.flatten(), yv.flatten()), axis=1)
                 xs = grid
                 xt = grid
            # pick a random metric
            metric = np.random.choice(possible_metrics)
            # one-hot encode the chosen metric
            #print("metric:", metric)
            metric_one_hot = [1.0 if m == metric else 0.0 for m in possible_metrics]
            C = ot.dist(xs, xt, metric=metric, use_tensor=True, p=3)
            C /= C.max()

        # Compute ground truth OT plan using POT
        otplan_solve = ot.solve(
            C, a=source_dist[i], b=target_dist[i]
        )
        # Sparse plan
        # otplan_emd_sparse = ot.emd_1d(
        #     xs, xt,
        #     source_dist[i], target_dist[i], dense=False
        # )
        otplan_solve_sparse = otplan_solve.sparse_plan

        # Dense plan
        # otplan_emd_dense = ot.emd_1d(
        #     xs, xt,
        #     source_dist[i], target_dist[i], dense=True
        # )
        otplan_solve_dense = otplan_solve.plan

        # Convert source/target distributions to tensors
        source_tensor = torch.tensor(source_dist[i], dtype=my_dtype_torch)
        target_tensor = torch.tensor(target_dist[i], dtype=my_dtype_torch)

        # Yield individual bin samples
        if row_wise:
            for j in range(n_bins):
                input_idx = [j / n_bins]

                input_total = torch.hstack((
                    source_tensor,
                    target_tensor,
                    torch.tensor(input_idx, dtype=my_dtype_torch)
                )).to(my_dtype_torch).to(device)

                # Build dict in the desired key order
                sample = {
                    'source_dist':  source_tensor.tolist(),
                    'target_dist':  target_tensor.tolist(),
                }
                # For semivariable, insert input_cost *before* input_cost_matrix
                #if cost_type == 'semivariable':
                sample['input_cost']  = metric_one_hot
                sample['input_cost_matrix'] = C.tolist()
                sample['input_idx']   = input_idx
                sample['input_total'] = input_total.tolist()
                # sample['target_otplan_emd_sparse'] = coo_tolist(otplan_emd_sparse, j)
                # sample['target_otplan_emd_dense']  = otplan_emd_dense[j].tolist()
                sample['target_otplan_solve_dense']  = otplan_solve_dense[j].tolist()
                sample['target_otplan_solve_sparse'] = coo_tolist(otplan_solve_sparse, j)

                yield sample

        else:
            input_total = torch.hstack((
                source_tensor,
                target_tensor
                )).to(my_dtype_torch).to(device)
            yield {
                'source_dist':  source_tensor.tolist(),
                'target_dist':  target_tensor.tolist(),
                'input_cost_matrix':  C.tolist(),
                'input_total':  input_total.tolist(),
                #'target_otplan_emd_sparse':   coo_tolist(otplan_emd_sparse, None),
                #'target_otplan_emd_dense':    otplan_emd_dense.tolist(),
                'target_otplan_solve_dense':  otplan_solve_dense.tolist(),
                'target_otplan_solve_sparse': coo_tolist(otplan_solve_sparse, None)
            }

def create_dataset_metadata(n_bins: int, dataset_size: int) -> Dict:
    """Create metadata for the dataset."""
    return {
        "description": "Optimal Transport Dataset for Learning Sparse Transport Mappings",
        "version": "1.0.0",
        "n_bins": n_bins,
        "dataset_size": dataset_size,
        "features": {
            "source_dist": {
                "description": "Source probability distribution",
                "dtype": "float32",
                "shape": [n_bins]
            },
            "target_dist": {
                "description": "Target probability distribution", 
                "dtype": "float32",
                "shape": [n_bins]
            },
            "input_idx": {
                "description": "Normalized row index (0-1)",
                "dtype": "float32",
                "shape": [1]
            },
            "total_input": {
                "description": "Concatenated input: source + target + row_idx",
                "dtype": "float32", 
                "shape": [2 * n_bins + 1]
            },
            "output_map_sparse": {
                "description": "Sparse representation of OT map row as string",
                "dtype": "string"
            }
        },
        "task": "optimal_transport",
        "domain": "mathematics",
        "license": "MIT"
    }


def upload_to_huggingface(dataset_dict, repo_name: str, token: str = None, private: bool = False):
    """Upload dataset to Hugging Face Hub."""
    try:
        print(f"🚀 Uploading dataset to Hugging Face Hub: {repo_name}")
        dataset_dict.push_to_hub(
            repo_id=repo_name,
            token=token,
            private=private,
            commit_message="Upload Optimal Transport dataset"
        )
        print(f"✅ Dataset uploaded successfully to https://huggingface.co/datasets/{repo_name}")
        return True
    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        return False

def create_dataset_card(repo_name: str, n_bins: int, dataset_size: int, output_path: str = "README.md"):
    """Create a dataset card (README.md) for the Hugging Face dataset."""
    
    card_content = f"""---
license: mit
task_categories:
- other
language:
- en
tags:
- optimal-transport
- mathematics
- sparse-matrices
- machine-learning
size_categories:
- 1M<n<10M
---

# Optimal Transport Dataset

## Dataset Description

This dataset contains synthetic data for learning optimal transport (OT) mappings between probability distributions using sparse matrix representations. The dataset is designed for training machine learning models to predict optimal transport matrices from source and target probability distributions.

## Dataset Structure

Each sample in the dataset contains:

- **source_dist**: Source probability distribution over discrete bins (list of floats)
- **target_dist**: Target probability distribution over discrete bins (list of floats)  
- **input_idx**: Normalized row index [0,1] for which to predict the OT mapping (list with one float)
- **total_input**: Concatenated input features (source + target + row_idx) (list of floats)
- **output_map_sparse**: Sparse representation of the corresponding row in the OT mapping matrix as string "[col1,val1,col2,val2,...]"

## Dataset Statistics

- **Number of bins**: {n_bins}
- **Total samples**: {dataset_size:,}
- **Train samples**: {int(dataset_size * 0.8):,}
- **Test samples**: {int(dataset_size * 0.2):,}
- **Input dimension**: {2 * n_bins + 1}
- **Sparse format**: COO (Coordinate) format for efficient storage

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_name}")

# Access train and test splits
train_data = dataset['train']
test_data = dataset['test']

# Get a sample
sample = train_data[0]
print(f"Source distribution: {{sample['source_dist']}}")
print(f"Target distribution: {{sample['target_dist']}}")
print(f"Input index: {{sample['input_idx']}}")
print(f"Sparse OT row: {{sample['output_map_sparse']}}")
```

## Sparse Format

The sparse OT mapping is stored as a string in the format:
```
[col1,val1,col2,val2,col3,val3,...]
```

Where:
- `col1, col2, col3, ...` are column indices (integers)
- `val1, val2, val3, ...` are corresponding values (floats with 3 decimal precision)

## Data Generation

The dataset is generated by:

1. Creating random source and target probability distributions over {n_bins} bins
2. Computing optimal transport mappings using the Python Optimal Transport (POT) library
3. Converting dense OT matrices to sparse COO format for efficient storage
4. Splitting the data into train/test sets (80/20 split)

## Applications

This dataset can be used for:

- Training neural networks to predict optimal transport mappings
- Research in optimal transport theory and applications
- Developing sparse matrix learning algorithms
- Benchmarking machine learning models on mathematical optimization tasks

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{{optimal_transport_dataset_2024,
  title={{Optimal Transport Dataset for Learning Sparse Transport Mappings}},
  author={{Your Name}},
  year={{2024}},
  url={{https://huggingface.co/datasets/{repo_name}}}
}}
```

## License

This dataset is released under the MIT License.
"""
    
    with open(output_path, 'w') as f:
        f.write(card_content)
    
    print(f"✅ Dataset card created: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create Hugging Face dataset for Optimal Transport")
    parser.add_argument("--n_bins", type=int, default=10, help="Number of bins for discretization")
    parser.add_argument("--dataset_size", type=int, default=1_000_000, help="Number of samples to generate")
    parser.add_argument("--row_wise", action="store_true", help="Store output plan matrix row by row?")
    parser.add_argument("--dim", type=int, default=1, help="Dimension of the distribution (1 or 2)")
    parser.add_argument("--output_dir", type=str, default="./ot_dataset", help="Output directory for dataset")
    parser.add_argument("--metric", type=str, default="sqeuclidean", help="Distance metric")
    parser.add_argument("--cost_fixed", action="store_true", help="Is the cost matrix C, fixed?")
    parser.add_argument("--cost_variable", action="store_true", help="Is the cost matrix C, variable?")
    parser.add_argument("--cost_semivariable", action="store_true", help="Is the cost matrix C, semivariable?")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--test_split", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random_split", action="store_true", help="Use random split instead of consecutive rows")
    parser.add_argument("--upload", action="store_true", help="Upload dataset to Hugging Face Hub")
    parser.add_argument("--repo_name", type=str, help="Repository name for Hugging Face Hub (e.g., 'username/ot-dataset')")
    parser.add_argument("--token", type=str, help="Hugging Face token (if not provided, will use token from huggingface_hub login)")
    parser.add_argument("--private", action="store_true", help="Make the repository private")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating dataset with {args.n_bins} bins and {args.dataset_size} samples")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    
    # Create dataset using generator
    print("Creating Hugging Face dataset...")
    if args.cost_fixed:
        cost_type = "fixed"
    elif args.cost_variable:
        cost_type = "variable"
    elif args.cost_semivariable:
        cost_type = "semivariable"
    else:
        raise ValueError("Invalid cost type")
    print(f"Cost type: {cost_type}")
    dataset = Dataset.from_generator(
        lambda: ot_sample_generator(args.n_bins, args.dataset_size, args.metric, cost_type, args.row_wise, args.device, args.dim)
    )

    # Split into train/test
    print(f"Splitting dataset with {args.test_split} test ratio using consecutive rows...")

    # Calculate split indices for consecutive rows
    total_samples = len(dataset)
    test_size = int(total_samples * args.test_split)
    train_size = total_samples - test_size

    # Take first train_size samples for training, last test_size for testing
    train_indices = list(range(train_size))
    test_indices  = list(range(train_size, total_samples))

    # Create train and test datasets
    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)

    # Create dataset dict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    # Add metadata
    metadata = create_dataset_metadata(args.n_bins, args.dataset_size)
    # Note: DatasetDict doesn't have info attribute, metadata is saved separately

    # Save dataset
    print(f"Saving dataset to {output_dir}...")
    dataset_dict.save_to_disk(str(output_dir))

    # Save metadata separately
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Dataset saved successfully!")
    print(f"Train samples: {len(dataset_dict['train'])}")
    print(f"Test samples: {len(dataset_dict['test'])}")
    print(f"Total samples: {len(dataset_dict['train']) + len(dataset_dict['test'])}")

    # Print sample
    print("\nSample from dataset:")
    sample = dataset_dict['train'][0]
    print(f"Source dist shape: {len(sample['source_dist'])}")
    print(f"Target dist shape: {len(sample['target_dist'])}")
    if args.row_wise:
        print(f"Input idx: {sample['input_idx']}")
    print(f"Input total shape: {len(sample['input_total'])}")
    #print(f"Output map sparse: {sample['output_map_sparse'][:100]}...")

    # Upload to Hugging Face Hub if requested
    if args.upload:
        if not args.repo_name:
            print("❌ Error: --repo_name is required when using --upload")
            print("Example: --repo_name 'username/ot-dataset'")
            return

        # Create dataset card
        card_path = output_dir / "README.md"
        create_dataset_card(args.repo_name, args.n_bins, args.dataset_size, str(card_path))

        # Upload dataset
        success = upload_to_huggingface(
            dataset_dict=dataset_dict,
            repo_name=args.repo_name,
            token=args.token,
            private=args.private
        )

        if success:
            print(f"\n🎉 Dataset uploaded successfully!")
            print(f"🔗 View at: https://huggingface.co/datasets/{args.repo_name}")
        else:
            print("\n❌ Upload failed!")
    else:
        print(f"\n💡 To upload to Hugging Face Hub, use:")
        print(f"   python3 create_hf_dataset.py --upload --repo_name 'username/ot-dataset'")

if __name__ == "__main__":
    main()