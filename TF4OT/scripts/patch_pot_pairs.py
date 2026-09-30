import sys

with open("TF4OT/run_inference_pot.py", "r") as f:
    code = f.read()

# We need to target the eval_all_1d_distribution_pairs code block
# Let's split by function definition to ensure we only patch eval_all_1d_distribution_pairs
parts = code.split('def eval_all_testset')
if len(parts) != 2:
    print("Error splitting")
    sys.exit(1)
    
pairs_code = parts[0]
testset_code = "def eval_all_testset" + parts[1]

# Apply patches to pairs_code

# Block 1
pairs_code = pairs_code.replace(
    "results_pot = {'mae': [], 'mape': [], 'hellinger': [], 'otcost_diff': [], 'tv': [], 'time_pot': []}",
    "results_pot = {'mae': [], 'mape': [], 'hellinger': [], 'otcost_diff': [], 'tv': [], 'time_pot': []}\n    results_pot_warm = {'mae': [], 'mape': [], 'hellinger': [], 'otcost_diff': [], 'tv': [], 'time_pot_warm': []}"
)

# Block 2
orig_solve = """                # --- Run POT exact OT solver ---
                import ot
                start_time_pot = time.time()
                try:
                    ot_pred_dense_pot_np = ot.emd(source_hist.cpu().numpy().astype(np.float64), 
                                                target_hist.cpu().numpy().astype(np.float64), 
                                                C.cpu().numpy().astype(np.float64))
                except Exception as e:
                    s_np = source_hist.cpu().numpy().astype(np.float64)
                    t_np = target_hist.cpu().numpy().astype(np.float64)
                    ot_pred_dense_pot_np = ot.emd(s_np / s_np.sum(), t_np / t_np.sum(), C.cpu().numpy().astype(np.float64))
                
                time_pot = time.time() - start_time_pot
                ot_pred_sparse_pot = sparse.coo_matrix(ot_pred_dense_pot_np)"""

new_solve = """                # --- Run POT exact OT solver ---
                import ot
                start_time_pot = time.time()
                try:
                    res = ot.solve(C.cpu().numpy().astype(np.float64), 
                                   a=source_hist.cpu().numpy().astype(np.float64), 
                                   b=target_hist.cpu().numpy().astype(np.float64))
                    ot_pred_dense_pot_np = res.plan
                except Exception as e:
                    s_np = source_hist.cpu().numpy().astype(np.float64)
                    t_np = target_hist.cpu().numpy().astype(np.float64)
                    res = ot.solve(C.cpu().numpy().astype(np.float64), a=s_np / s_np.sum(), b=t_np / t_np.sum())
                    ot_pred_dense_pot_np = res.plan
                
                time_pot = time.time() - start_time_pot
                ot_pred_sparse_pot = sparse.coo_matrix(ot_pred_dense_pot_np)
                
                # --- Run POT Warm (using ToR-OT as init) ---
                start_time_pot_warm = time.time()
                try:
                    res_w = ot.solve(C.cpu().numpy().astype(np.float64), 
                                   a=source_hist.cpu().numpy().astype(np.float64), 
                                   b=target_hist.cpu().numpy().astype(np.float64),
                                   plan_init=ot_pred_dense.cpu().numpy().astype(np.float64))
                    ot_pred_dense_pot_warm_np = res_w.plan
                except Exception as e:
                    s_np = source_hist.cpu().numpy().astype(np.float64)
                    t_np = target_hist.cpu().numpy().astype(np.float64)
                    res_w = ot.solve(C.cpu().numpy().astype(np.float64), a=s_np / s_np.sum(), b=t_np / t_np.sum(), plan_init=ot_pred_dense.cpu().numpy().astype(np.float64))
                    ot_pred_dense_pot_warm_np = res_w.plan
                
                time_pot_warm = time.time() - start_time_pot_warm
                ot_pred_sparse_pot_warm = sparse.coo_matrix(ot_pred_dense_pot_warm_np)"""

pairs_code = pairs_code.replace(orig_solve, new_solve)

# Block 3: POT Metrics
orig_metrics = """                # --- Metrics for POT ---
                mae_pot  = utils.sparse_mean_abs_error(ot_target_sparse, ot_pred_sparse_pot)
                mape_pot = utils.sparse_weighted_mean_abs_percentage_error(ot_target_sparse, ot_pred_sparse_pot)
                hellinger_pot = utils.sparse_hellinger_distance(ot_target_sparse, ot_pred_sparse_pot)
                tv_pot = utils.sparse_total_variation(ot_target_sparse, ot_pred_sparse_pot)
                otcost_pred_pot = (ot_pred_dense_pot_np * C.cpu().numpy()).sum().item()
                otcost_diff_pot = abs(otcost_pred_pot - otcost_gt)"""

new_metrics = """                # --- Metrics for POT ---
                mae_pot  = utils.sparse_mean_abs_error(ot_target_sparse, ot_pred_sparse_pot)
                mape_pot = utils.sparse_weighted_mean_abs_percentage_error(ot_target_sparse, ot_pred_sparse_pot)
                hellinger_pot = utils.sparse_hellinger_distance(ot_target_sparse, ot_pred_sparse_pot)
                tv_pot = utils.sparse_total_variation(ot_target_sparse, ot_pred_sparse_pot)
                otcost_pred_pot = (ot_pred_dense_pot_np * C.cpu().numpy()).sum().item()
                otcost_diff_pot = abs(otcost_pred_pot - otcost_gt)
                
                # --- Metrics for POT Warm ---
                mae_pot_warm  = utils.sparse_mean_abs_error(ot_target_sparse, ot_pred_sparse_pot_warm)
                mape_pot_warm = utils.sparse_weighted_mean_abs_percentage_error(ot_target_sparse, ot_pred_sparse_pot_warm)
                hellinger_pot_warm = utils.sparse_hellinger_distance(ot_target_sparse, ot_pred_sparse_pot_warm)
                tv_pot_warm = utils.sparse_total_variation(ot_target_sparse, ot_pred_sparse_pot_warm)
                otcost_pred_pot_warm = (ot_pred_dense_pot_warm_np * C.cpu().numpy()).sum().item()
                otcost_diff_pot_warm = abs(otcost_pred_pot_warm - otcost_gt)"""
pairs_code = pairs_code.replace(orig_metrics, new_metrics)

# Block 4: Appending stats
orig_append = """                # Collect metrics for POT
                results_pot['mae'].append(mae_pot)
                results_pot['mape'].append(mape_pot)
                results_pot['hellinger'].append(hellinger_pot)
                results_pot['tv'].append(tv_pot)
                results_pot['otcost_diff'].append(otcost_diff_pot)
                results_pot['time_pot'].append(time_pot)"""

new_append = """                # Collect metrics for POT
                results_pot['mae'].append(mae_pot)
                results_pot['mape'].append(mape_pot)
                results_pot['hellinger'].append(hellinger_pot)
                results_pot['tv'].append(tv_pot)
                results_pot['otcost_diff'].append(otcost_diff_pot)
                results_pot['time_pot'].append(time_pot)
                
                results_pot_warm['mae'].append(mae_pot_warm)
                results_pot_warm['mape'].append(mape_pot_warm)
                results_pot_warm['hellinger'].append(hellinger_pot_warm)
                results_pot_warm['tv'].append(tv_pot_warm)
                results_pot_warm['otcost_diff'].append(otcost_diff_pot_warm)
                results_pot_warm['time_pot_warm'].append(time_pot_warm)"""
pairs_code = pairs_code.replace(orig_append, new_append)

# Block 5: Row_data dictionaries (to export to .npy)
orig_row = """                    'time_pot': time_pot,
                    'mae_pot': mae_pot,
                    'otcost_diff_pot': otcost_diff_pot
                }"""
new_row = """                    'time_pot': time_pot,
                    'mae_pot': mae_pot,
                    'otcost_diff_pot': otcost_diff_pot,
                    'time_pot_warm': time_pot_warm,
                    'mae_pot_warm': mae_pot_warm,
                    'otcost_diff_pot_warm': otcost_diff_pot_warm
                }"""
pairs_code = pairs_code.replace(orig_row, new_row)

final_code = pairs_code + testset_code

with open("TF4OT/run_inference_pot.py", "w") as f:
    f.write(final_code)

print("Patch applied to OOD Pairs!")
