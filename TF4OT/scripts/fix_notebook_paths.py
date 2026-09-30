import nbformat
import os

notebook_path = "TF4OT/evals/plots/comparison_box_plots_1d.ipynb"

def fix_notebook(nb_path):
    abs_path = os.path.abspath(nb_path)
    if not os.path.exists(abs_path):
        print(f"Skipping {nb_path} (not found)")
        return

    print(f"Processing {nb_path}...")
    with open(abs_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    changes_made = False
    for cell in nb.cells:
        if cell.cell_type == 'code':
            source = cell.source
            
            # We want to replace "eval_all_testset" with "eval_all_pairs"
            # AND replace ".npz" with ".npy"
            # AND revert the .item() call I added if present.
            
            # The user code snippet:
            # path_tot = os.path.join(dir_joint, f"eval_all_testset_1d_{metric}_results_tot.npz")
            # should become
            # path_tot = os.path.join(dir_joint, f"eval_all_pairs_1d_{metric}_results_tot.npy")
            
            # And similarly for path_not.
            
            if "eval_all_testset_1d" in source:
                new_source = source.replace("eval_all_testset_1d", "eval_all_pairs_1d")
                new_source = new_source.replace(".npz", ".npy")
                
                # Also revert the .item() loading if I added it or if it's there
                # My previous fix might currently be:
                # data_tot = np.load(path_tot, allow_pickle=True)['results_by_metric'].item()
                # If we switch to .npy, we just need:
                # data_tot = np.load(path_tot, allow_pickle=True)
                
                # Let's clean up the loading lines
                lines = new_source.split('\n')
                fixed_lines = []
                for line in lines:
                    if "np.load(path_tot" in line or "np.load(path_not" in line:
                        # Normalize to simple load
                        # Keep indentation
                        indent = line[:len(line) - len(line.lstrip())]
                        variable = line.strip().split('=')[0].strip() # data_tot or data_not
                        variable_path = "path_tot" if "path_tot" in line else "path_not"
                        
                        fixed_line = f"{indent}{variable} = np.load({variable_path}, allow_pickle=True)"
                        fixed_lines.append(fixed_line)
                    else:
                        fixed_lines.append(line)
                
                new_source = '\n'.join(fixed_lines)
                
                if new_source != source:
                    cell.source = new_source
                    changes_made = True
                    print("  Applied path and load fix.")

    if changes_made:
        with open(abs_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print("Saved changes.")
    else:
        print("No changes made.")

if __name__ == "__main__":
    fix_notebook(notebook_path)
