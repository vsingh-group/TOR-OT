import numpy as np
import os

# Paths mentioned in the notebook (via view_file earlier)
# ../../../experiments/cost_fixed/model_...
# I need to translate this relative path to absolute or find where it is locally.
# Based on generate_plots.py, it seems to assume a structure.
# Let's try to find one of the npy files using fd first to be sure of the path.

base_path = "/home/ivan/repos/TF4OT/experiments" 
# Assuming experiments dir is at repo root or similar. 
# Actually, generate_plots.py uses "TF4OT/evals/plots/" as current dir relative to root? No, run_command was in root.
# generate_plots.py has:
# EXPERIMENT_DIR_INDIV = "experiments/cost_fixed/"
# Let's check if that exists.

npy_file = "experiments/cost_fixed/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_cityblock/eval_all_pairs_1d_cityblock_results_tot.npy"


METRICS = ['cityblock', 'euclidean', 'sqeuclidean', 'minkowski', 'chebyshev', 'canberra'] # From notebook

base_dir = "experiments/cost_fixed"

# Find the specific model directory
model_dir = None
for item in os.listdir(base_dir):
    if item.startswith("model_dim768") and "100K" in item:
        model_dir = os.path.join(base_dir, item)
        break

if not model_dir:
    print("Could not find model directory.")
else:
    for metric in METRICS:
        # Construct path similar to notebook logic, though notebook hardcodes paths somewhat.
        # But wait, looking at file output from earlier, the directory name CHANGES with metric?
        # Yes: ..._cityblock/eval_...
        # So I need to find the dir for each metric.
        
        current_metric_dir = None
        for item in os.listdir(base_dir):
             if item.startswith("model_dim768") and item.endswith(f"_{metric}"):
                 current_metric_dir = os.path.join(base_dir, item)
                 break
        
        if not current_metric_dir:
            print(f"Skipping {metric}, dir not found.")
            continue

        npy_file = os.path.join(current_metric_dir, f"eval_all_pairs_1d_{metric}_results_tot.npy")
        if os.path.exists(npy_file):
            print(f"\nChecking {metric}: {npy_file}")
            try:
                data = np.load(npy_file, allow_pickle=True)
                print(f"  Shape: {data.shape}")
                
                # Check for non-dict items
                bad_items = []
                for i, item in enumerate(data):
                    if not isinstance(item, dict):
                        bad_items.append((i, type(item), item))
                
                if bad_items:
                    print(f"  FAILED: Found {len(bad_items)} non-dict items!")
                    print(f"  Example bad item: index {bad_items[0][0]}, type {bad_items[0][1]}, value: {bad_items[0][2]}")
                else:
                    print("  OK: All items are dicts.")
                    
            except Exception as e:
                print(f"  Error loading: {e}")


# Check .npz files in cost_semivariable
base_dir_joint = "experiments/cost_semivariable"
if os.path.exists(base_dir_joint):
    print(f"\nScanning {base_dir_joint} for .npz...")
    for root, dirs, files in os.walk(base_dir_joint):
        for file in files:
            if file.endswith("results_tot.npz"):
                npz_file = os.path.join(root, file)
                print(f"Checking: {npz_file}")
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    print(f"  Type: {type(data)}")
                    print(f"  Keys: {list(data.keys())}")
                    

                    if 'results_by_metric' in data:
                        rbm = data['results_by_metric']
                        print(f"  Shape of results_by_metric: {rbm.shape}")
                        if len(rbm) > 0:
                            print(f"  First item type: {type(rbm[0])}")
                            if isinstance(rbm[0], dict):
                                print(f"  First item keys: {rbm[0].keys()}")
                    else:
                        print("  'results_by_metric' key NOT found.")

                except Exception as e:
                    print(f"  Error loading: {e}")


# Check specific testset npz
testset_npz = "experiments/cost_semivariable/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data600K_rowwise_costsemivariable_prec4_bins30_DigitTokenizer_epochs25_batch128/eval_all_testset_1d_cityblock_results_tot.npz"

if os.path.exists(testset_npz):
    print(f"\nChecking SPECIFIC testset file: {testset_npz}")
    try:
        # User code: data_tot = np.load(path_tot, allow_pickle=True)['results_by_metric'].item()
        data = np.load(testset_npz, allow_pickle=True)
        if 'results_by_metric' in data:
            rbm_item = data['results_by_metric'].item()
            print(f"  Type of item(): {type(rbm_item)}")
            
            if isinstance(rbm_item, dict):
                print(f"  It is a DICT. Keys[:5]: {list(rbm_item.keys())[:5]}")
                print("  This confirms iterating over it yields keys (strings).")
                
                # We likely want the VALUES if it's a dict of results?
                # Or maybe it's a dict where one key holds the list?
                first_key = list(rbm_item.keys())[0]
                first_val = rbm_item[first_key]
                print(f"  Value for key '{first_key}': type={type(first_val)}")
                print(f"  Value sample: {first_val}")

            elif isinstance(rbm_item, list):
                print(f"  It is a LIST. Length: {len(rbm_item)}")
                if len(rbm_item) > 0:
                     print(f"  Element 0: {rbm_item[0]}")
            
        else:
            print("  'results_by_metric' key NOT found.")
            
    except Exception as e:
        print(f"  Error loading: {e}")

# Check 2D testset npy
testset_2d_npy = "experiments/cost_fixed/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins36_DigitTokenizer_epochs50_batch192_sqeuclidean/eval_all_testset_2d_sqeuclidean_results_tot.npy"

if os.path.exists(testset_2d_npy):
    print(f"\nChecking 2D testset file: {testset_2d_npy}")
    try:
        data = np.load(testset_2d_npy, allow_pickle=True)
        print(f"  Type: {type(data)}")
        if isinstance(data, np.ndarray):
             print(f"  Shape: {data.shape}")
             if data.size > 0:
                 item = data.item() if data.ndim == 0 else data[0]
                 print(f"  Item type: {type(item)}")
                 if isinstance(item, dict):
                     print(f"  Keys: {list(item.keys())}")
    except Exception as e:
        print(f"  Error loading: {e}")

