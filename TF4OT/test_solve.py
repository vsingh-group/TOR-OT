import ot
import numpy as np
import time

np.random.seed(42)
N = 100
a = np.random.rand(N)
a /= a.sum()
b = np.random.rand(N)
b /= b.sum()
C = np.random.rand(N, N)

# Exact dense
t0 = time.time()
res_exact = ot.solve(C, a=a, b=b)
P_exact = res_exact.plan
t1 = time.time()

# Let's say our prediction gets exactly the plan (or a very close one)
mask = P_exact > 1e-8
# We need a dense matrix for plan_init
P_init = np.zeros_like(P_exact)
P_init[mask] = P_exact[mask]

# Warm start
t2 = time.time()
res_warm = ot.solve(C, a=a, b=b, plan_init=P_init)
P_warm = res_warm.plan
t3 = time.time()

print(f"Exact Time: {t1 - t0:.5f}s, sum: {P_exact.sum()}")
print(f"Warm Time:  {t3 - t2:.5f}s, sum: {P_warm.sum()}")
print(f"Difference in plan: {np.abs(P_exact - P_warm).max():.5e}")
