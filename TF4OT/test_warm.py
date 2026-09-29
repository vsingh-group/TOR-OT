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

# Exact
t0 = time.time()
P_exact = ot.emd(a, b, C)
t1 = time.time()

# Let's say our "prediction" gets the exact support + a little noise
mask = P_exact > 1e-8
C_warm = C.copy()
C_warm[~mask] = C.max() * 1e5

t2 = time.time()
P_warm = ot.emd(a, b, C_warm)
t3 = time.time()

print(f"Exact Time: {t1 - t0:.5f}s, sum: {P_exact.sum()}")
print(f"Warm Time:  {t3 - t2:.5f}s, sum: {P_warm.sum()}")
print(f"Difference in plan: {np.abs(P_exact - P_warm).max():.5e}")
