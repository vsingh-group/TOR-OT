import time
import numpy as np
import ot

def apply_sinkhorn_projection(ot_pred_dense, source_pdf, target_pdf, reg=1.0, numItermax=500):
    projected = np.zeros_like(ot_pred_dense)
    b_size = ot_pred_dense.shape[0]
    for i in range(b_size):
        # We model the negative log probability as cost to feed into standard sinkhorn
        P_pred = np.clip(ot_pred_dense[i], 1e-9, 1.0)
        M_cost = -np.log(P_pred)
        
        a = source_pdf
        b = target_pdf
        try:
            P_proj = ot.sinkhorn(a, b, M_cost, reg, numItermax=numItermax, stopThr=1e-5)
            projected[i] = P_proj
        except Warning:
            projected[i] = P_pred
    return projected

B = 100
N = 30
M = 30
ot_pred_dense = np.random.rand(B, N, M)
# Normalize rows
ot_pred_dense = ot_pred_dense / ot_pred_dense.sum(axis=-1, keepdims=True)

source = np.ones(N) / N
target = np.ones(M) / M

start = time.time()
proj = apply_sinkhorn_projection(ot_pred_dense, source, target)
print(f"Avg time per item (N=30x30): {(time.time() - start) / B * 1000:.3f} ms")
