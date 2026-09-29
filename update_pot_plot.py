import sys

with open("TF4OT/compare_pot_tot.py", "r") as f:
    code = f.read()

# Add Warmstart to table headers
if "ToR-OT (Sym)" in code and "POT Warm" not in code:
    code = code.replace(
        "| Metric    | POT (exact)   |   TOT (Ours) |   ToR-OT (Sym) | NeuralOT   |",
        "| Metric    | POT (exact)   | POT Warmstart |   TOT (Ours) |   ToR-OT (Sym) | NeuralOT   |"
    )
    code = code.replace(
        "| Metric    | POT (exact)   |   TOT (Ours) | ToR-OT (Sym)   | NeuralOT   |",
        "| Metric    | POT (exact)   | POT Warmstart |   TOT (Ours) | ToR-OT (Sym)   | NeuralOT   |"
    )

# Inject ts_val logic for POT Warm
if 'ts_results["POT"] = ' in code and 'ts_results["POT (Warm)"]' not in code:
    orig = 'ts_results["POT"] = {name: ts_val(ts_pot, k_pot) if k_pot else None\n                     for name, _, k_pot, _ in TESTSET_ACC}'
    new = orig + '\n\nts_results["POT (Warm)"] = {name: ts_val(ts_pot, k_pot + "_warm") if k_pot else None\n                            for name, _, k_pot, _ in TESTSET_ACC}'
    code = code.replace(orig, new)
    code = code.replace('ts_results["POT"]["Max Viol."] = None', 'ts_results["POT"]["Max Viol."] = None\nts_results["POT (Warm)"]["Max Viol."] = None')
    
    code = code.replace('ts_results["POT"]["Time (s)"]    = time_pot_ts', 'ts_results["POT"]["Time (s)"]    = time_pot_ts\nts_results["POT (Warm)"]["Time (s)"] = ts_val(ts_pot, "all_time_pot_warm")')
    code = code.replace('+===========+===============+==============+================+============+', '+===========+===============+===============+==============+================+============+')
    code = code.replace('row = [metric, fmt(ts_results["POT"][metric]),', 'row = [metric, fmt(ts_results["POT"][metric]), fmt(ts_results["POT (Warm)"][metric]),')
    code = code.replace('ts_results["TOT"][metric]),', 'ts_results["TOT"][metric]),') # Keep untouched
    

if 'ood_results["POT"] = ' in code and 'ood_results["POT (Warm)"]' not in code:
    orig2 = 'ood_results["POT"] = {m: None for m in [n for n, *_ in OOD_ACC]}'
    new2 = orig2 + '\nood_results["POT (Warm)"] = {m: None for m in [n for n, *_ in OOD_ACC]}'
    code = code.replace(orig2, new2)
    
    orig3 = 'ood_results["POT"]["Time (s)"]    = ood_val("time_pot")'
    new3 = orig3 + '\nood_results["POT (Warm)"]["Time (s)"] = ood_val("time_pot_warm")'
    code = code.replace(orig3, new3)

if 'row2 = [metric, fmt(ood_results["POT"][metric]),' in code:
    code = code.replace('row2 = [metric, fmt(ood_results["POT"][metric]),', 'row2 = [metric, fmt(ood_results["POT"][metric]), fmt(ood_results["POT (Warm)"][metric]),')

with open("TF4OT/compare_pot_tot.py", "w") as f:
    f.write(code)

print("Plotting script updated!")
