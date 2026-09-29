import nbformat
import os

notebooks = [
    "TF4OT/evals/plots/comparison_box_plots_1d.ipynb",
    "TF4OT/evals/plots/comparison_box_plots_2d.ipynb"
]

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
            # Look for context where .npz is used
            if ".npz" in source and "np.load(path_tot, allow_pickle=True)" in source:
                print("Found match in cell.")
                # We want to replace the load call, but ONLY if we are sure it corresponds to the .npz path.
                # Since .npz is in the source, it's highly likely.
                # However, strictly speaking, path_tot might be re-assigned.
                # Given the snippet, it's:
                # path_tot = ... .npz
                # data_tot = np.load(path_tot, ...)
                
                old_line = "data_tot = np.load(path_tot, allow_pickle=True)"
                new_line = "data_tot = np.load(path_tot, allow_pickle=True)['results_by_metric'].item()"
                
                if old_line in source:
                    new_source = source.replace(old_line, new_line)
                    cell.source = new_source
                    changes_made = True
                    print("  Applied fix.")
                else:
                    print("  Could not find exact load line to replace.")
            
    if changes_made:
        with open(abs_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print("Saved changes.")
    else:
        print("No changes made.")

if __name__ == "__main__":
    for nb in notebooks:
        fix_notebook(nb)
