import sys
import os

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Find the statistics printing part
    targetStr = """    logger.info("OT Cost Difference:")"""
    
    if "Constraint Violations (Row):" not in content and targetStr in content:
        insertStr = """
    logger.info("Constraint Violations (Row):")
    logger.info(f"  Mean:   {all_row_violation.mean():.6e}")
    logger.info(f"  Std:    {all_row_violation.std():.6e}")
    logger.info("="*80 + "\\n")
    
    logger.info("Constraint Violations (Col):")
    logger.info(f"  Mean:   {all_col_violation.mean():.6e}")
    logger.info(f"  Std:    {all_col_violation.std():.6e}")
    logger.info("="*80 + "\\n")
    
"""
        content = content.replace(targetStr, insertStr + targetStr)
        
        # Now update the table logic
        if "table_data.append([" in content:
            # We want to add row and col violation to the table
            table_append = "table_data.append([\\n            r['batch_idx'],\\n            r.get('metric', 'N/A'),\\n            f\"{r['mae_sparse']:.3e}\",\\n            f\"{r['mape_sparse']:.3e}\",\\n            f\"{r['hellinger_sparse']:.3e}\",\\n            f\"{r['tv_sparse']:.3e}\",\\n            f\"{r['otcost_diff']:.3e}\",\\n            f\"{r['row_violation']:.3e}\",\\n            f\"{r['col_violation']:.3e}\"\\n        ])"
            
            content = content.replace("table_data.append([\n            r['batch_idx'],\n            r.get('metric', 'N/A'),\n            f\"{r['mae_sparse']:.3e}\",\n            f\"{r['mape_sparse']:.3e}\",\n            f\"{r['hellinger_sparse']:.3e}\",\n            f\"{r['tv_sparse']:.3e}\",\n            f\"{r['otcost_diff']:.3e}\"\n        ])", table_append)
        
        if "headers = ['Batch', 'Metric', 'MAE', 'MAPE', 'Hellinger', 'TV', 'OT Cost Diff']" in content:
            content = content.replace("headers = ['Batch', 'Metric', 'MAE', 'MAPE', 'Hellinger', 'TV', 'OT Cost Diff']", "headers = ['Batch', 'Metric', 'MAE', 'MAPE', 'Hellinger', 'TV', 'OT Cost Diff', 'Row Violation', 'Col Violation']")

        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Patched {filepath} successfully.")
    else:
        print(f"Could not patch {filepath} (maybe already patched?)")

patch_file("run_inference.py")
patch_file("run_inference_sym.py")
