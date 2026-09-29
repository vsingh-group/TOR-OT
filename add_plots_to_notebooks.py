import nbformat
import os

notebooks = [
    "TF4OT/evals/plots/comparison_box_plots_1d.ipynb",
    "TF4OT/evals/plots/comparison_box_plots_2d.ipynb"
]

code_content = r"""
# -----------------------------
# Amortized Runtime Comparison
# -----------------------------
print(f"\n{'='*40}\nGenerating Amortized Runtime Plot\n{'='*40}")

T_TRAIN_HOURS = 72
T_TRAIN_SEC = T_TRAIN_HOURS * 3600

# User specified T_infer=1s for amortized method (TOT)
T_TOT_INFER_SEC = 1.0 

# Estimated runtime for Neural OT (NOT) based on max_steps=2501 in toy code.
# Assuming ~10-15ms per step => ~30 seconds per sample.
T_NOT_SEC = 2500.0 

# Determine break-even N
# T_TRAIN/N + T_TOT_INFER = T_NOT
if T_NOT_SEC > T_TOT_INFER_SEC:
    N_break_even = T_TRAIN_SEC / (T_NOT_SEC - T_TOT_INFER_SEC)
else:
    N_break_even = 1e6 # Fallback

# --- Plot 1: Log Scale ---
min_exp = 3  # 10^3
max_exp = int(np.log10(N_break_even)) + 2 
N_values_log = np.logspace(min_exp, max_exp, 100)

avg_time_tot_log = (T_TRAIN_SEC / N_values_log) + T_TOT_INFER_SEC
avg_time_not_log = np.full_like(N_values_log, T_NOT_SEC)

plt.figure(figsize=(8, 6))
plt.loglog(N_values_log, avg_time_not_log, label=f'Standard (NOT): {T_NOT_SEC}s/sample', linewidth=2, color='tab:orange', linestyle='--')
plt.loglog(N_values_log, avg_time_tot_log, label=f'Amortized (TOT): {T_TRAIN_HOURS}h train + {T_TOT_INFER_SEC}s/sample', linewidth=2, color='tab:blue')
plt.plot(N_break_even, T_NOT_SEC, 'ko', label=f'Break-even: N $\\approx$ {int(N_break_even):,}')

plt.xlabel('Number of Samples (N) [Log scale]', fontsize=14)
plt.ylabel('Average Time per Sample (Seconds) [Log scale]', fontsize=14)
plt.title('Amortized Runtime Comparison (Log Scale)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.grid(True, which="major", ls="-", alpha=0.5)
plt.tight_layout()
plt.show()


# --- Plot 2: Linear Scale ---
max_N = int(N_break_even * 4) 
N_values_lin = np.linspace(100, max_N, 500)

avg_time_tot_lin = (T_TRAIN_SEC / N_values_lin) + T_TOT_INFER_SEC
avg_time_not_lin = np.full_like(N_values_lin, T_NOT_SEC)

plt.figure(figsize=(8, 6))
plt.plot(N_values_lin, avg_time_not_lin, label=f'Standard (NOT): {T_NOT_SEC}s/sample', linewidth=2, color='tab:orange', linestyle='--')
plt.plot(N_values_lin, avg_time_tot_lin, label=f'Amortized (TOT): {T_TRAIN_HOURS}h train + {T_TOT_INFER_SEC}s/sample', linewidth=2, color='tab:blue')
plt.plot(N_break_even, T_NOT_SEC, 'ko', label=f'Break-even: N $\\approx$ {int(N_break_even):,}')

plt.xlabel('Number of Samples (N)', fontsize=14)
plt.ylabel('Average Time per Sample (Seconds)', fontsize=14)
plt.title('Amortized Runtime Comparison (Linear Scale)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.grid(True, which="major", ls="-", alpha=0.5)
plt.tight_layout()
plt.show()
"""

for nb_path in notebooks:
    abs_path = os.path.abspath(nb_path)
    if not os.path.exists(abs_path):
        print(f"Error: {abs_path} does not exist.")
        continue
    
    print(f"Reading {abs_path}...")
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Check if cell already exists to avoid duplicates
        cell_exists = False
        for cell in nb.cells:
            if "Amortized Runtime Comparison" in cell.source:
                cell_exists = True
                break
        
        if not cell_exists:
            new_cell = nbformat.v4.new_code_cell(code_content)
            nb.cells.append(new_cell)
            print("Appending new cell...")
            
            with open(abs_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            print("Done.")
        else:
            print("Cell already exists. Skipping.")
            
    except Exception as e:
        print(f"Failed to process {abs_path}: {e}")
