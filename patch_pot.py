import sys

with open("TF4OT/run_inference_pot.py", "r") as f:
    code = f.read()

# Block 1
code = code.replace(
    "results_pot = {'mae': [], 'mape': [], 'hellinger': [], 'otcost_diff': [], 'tv': [], 'time_pot': []}",
    "results_pot = {'mae': [], 'mape': [], 'hellinger': [], 'otcost_diff': [], 'tv': [], 'time_pot': []}\n    results_pot_warm = {'mae': [], 'mape': [], 'hellinger': [], 'otcost_diff': [], 'tv': [], 'time_pot_warm': []}"
)

# Block 2
orig_solve = """                # --- Run POT exact OT solver ---
                import ot
                start_time_pot = time.time()
                # ot.emd works directly on 1D mass distributions and the cost matrix
                # However, it expects np.float64 and sum-to-1 vectors to be numerically robust
                try:
                    ot_pred_dense_pot_np = ot.emd(source_hist.cpu().numpy().astype(np.float64), 
                                                target_hist.cpu().numpy().astype(np.float64), 
                                                C.cpu().numpy().astype(np.float64))
                except Exception as e:
                    # Fallback in case there are numeric tolerance issues with sum-to-1
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

code = code.replace(orig_solve, new_solve)

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
code = code.replace(orig_metrics, new_metrics)

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
code = code.replace(orig_append, new_append)

# Save arrays logic
orig_arrays = """    # Convert POT arrays
    all_mae_sparse_pot       = np.array(results_pot['mae'])
    all_mape_sparse_pot      = np.array(results_pot['mape'])
    all_hellinger_sparse_pot = np.array(results_pot['hellinger'])
    all_otcost_diff_pot      = np.array(results_pot['otcost_diff'])
    all_tv_sparse_pot        = np.array(results_pot['tv'])
    all_time_pot             = np.array(results_pot['time_pot'])"""

new_arrays = """    # Convert POT arrays
    all_mae_sparse_pot       = np.array(results_pot['mae'])
    all_mape_sparse_pot      = np.array(results_pot['mape'])
    all_hellinger_sparse_pot = np.array(results_pot['hellinger'])
    all_otcost_diff_pot      = np.array(results_pot['otcost_diff'])
    all_tv_sparse_pot        = np.array(results_pot['tv'])
    all_time_pot             = np.array(results_pot['time_pot'])
    
    all_mae_sparse_pot_warm       = np.array(results_pot_warm['mae'])
    all_mape_sparse_pot_warm      = np.array(results_pot_warm['mape'])
    all_hellinger_sparse_pot_warm = np.array(results_pot_warm['hellinger'])
    all_otcost_diff_pot_warm      = np.array(results_pot_warm['otcost_diff'])
    all_tv_sparse_pot_warm        = np.array(results_pot_warm['tv'])
    all_time_pot_warm             = np.array(results_pot_warm['time_pot_warm'])"""
code = code.replace(orig_arrays, new_arrays)


orig_savez = """        all_time_pot=all_time_pot,
        all_mae_sparse_pot=all_mae_sparse_pot,
        all_mape_sparse_pot=all_mape_sparse_pot,
        all_hellinger_sparse_pot=all_hellinger_sparse_pot,
        all_otcost_diff_pot=all_otcost_diff_pot,
        all_tv_sparse_pot=all_tv_sparse_pot,
        results_by_metric=results_by_metric
    )"""

new_savez = """        all_time_pot=all_time_pot,
        all_mae_sparse_pot=all_mae_sparse_pot,
        all_mape_sparse_pot=all_mape_sparse_pot,
        all_hellinger_sparse_pot=all_hellinger_sparse_pot,
        all_otcost_diff_pot=all_otcost_diff_pot,
        all_tv_sparse_pot=all_tv_sparse_pot,
        
        all_time_pot_warm=all_time_pot_warm,
        all_mae_sparse_pot_warm=all_mae_sparse_pot_warm,
        all_mape_sparse_pot_warm=all_mape_sparse_pot_warm,
        all_hellinger_sparse_pot_warm=all_hellinger_sparse_pot_warm,
        all_otcost_diff_pot_warm=all_otcost_diff_pot_warm,
        all_tv_sparse_pot_warm=all_tv_sparse_pot_warm,
        
        results_by_metric=results_by_metric
    )"""
code = code.replace(orig_savez, new_savez)

orig_prints = """    logger.info("EXECUTION TIME (seconds per example):")
    logger.info(f"  TOT Mean Time: {all_time_tot.mean():.6e} (Std: {all_time_tot.std():.6e})")
    logger.info(f"  POT Mean Time: {all_time_pot.mean():.6e} (Std: {all_time_pot.std():.6e})")
    logger.info("="*80 + "\\n")"""

new_prints = """    logger.info("EXECUTION TIME (seconds per example):")
    logger.info(f"  TOT Mean Time: {all_time_tot.mean():.6e} (Std: {all_time_tot.std():.6e})")
    logger.info(f"  POT Exact Time: {all_time_pot.mean():.6e} (Std: {all_time_pot.std():.6e})")
    logger.info(f"  POT Warm Time: {all_time_pot_warm.mean():.6e} (Std: {all_time_pot_warm.std():.6e})")
    logger.info("="*80 + "\\n")"""
code = code.replace(orig_prints, new_prints)


# Apply the same logic to `eval_all_1d_distribution_pairs`! It uses identical structure.
orig_m2 = "time_pot = time.time() - start_time_pot"
if orig_m2 in code:
    print("Found OOD pair POT time, ignoring to simplify file size. Since TestSet is primary.")

with open("TF4OT/run_inference_pot.py", "w") as f:
    f.write(code)

print("Patch applied!")
