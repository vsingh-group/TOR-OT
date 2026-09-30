import numpy as np
d = np.load("experiments/cost_fixed/model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_epochs50_batch192_sqeuclidean/eval_all_pairs_1d_sqeuclidean_results_pot.npy", allow_pickle=True)

time_pot = []
time_warm = []
mae_diff = []

for row in d:
    time_pot.append(row["time_pot"])
    time_warm.append(row["time_pot_warm"])
    mae_diff.append(abs(row["mae_pot"] - row["mae_pot_warm"]))

print(f"OOD Quick Test Results (144 pairs):")
print(f"POT Exact Time Average:      {np.mean(time_pot):.6f} sec")
print(f"POT Warmstart Time Average:  {np.mean(time_warm):.6f} sec")
print(f"POT Acceleration:            {np.mean(time_pot) / np.mean(time_warm):.1f}x faster")
print(f"MAE structural diff:         {np.max(mae_diff):.5e} (0 means Warmstart yielded identical exact mapping!)")