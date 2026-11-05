"""Classical baselines on the shared sets (CPU; run on a local machine, not the GPU server).

    python -m baselines.pot.solve <camera_ready dir> [workers]         (env_torot, repository root; CPU)

* pot             : exact network simplex (ot.emd) = the reference plans; only its time is new.
* sinkhorn_eps*   : entropic OT (log-domain Sinkhorn, POT), eps relative to the max-normalised cost, run until the
                    marginal error is below 1e-9 or 200,000 iterations. Zero-mass bins are removed before solving and
                    re-inserted. Per set: iteration statistics and the fraction of converged instances.
Each (set, method) runs in its own single-threaded process; times are per instance on one core.
Writes preds/<set>__<method>.npy (Sinkhorn plans) and preds/<set>__<method>.timing.json.
"""
import os
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"                     # one core per worker: comparable per-instance times

import glob, json, platform, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, ot
from torot.data.cost import cost_matrix
from torot.data.supports import set_coords

EPS = (1e-2, 1e-3)
STOP_THR, MAX_ITER = 1e-9, 200_000


def sinkhorn(a, b, C, eps):
    i, j = a > 0, b > 0
    P = np.zeros_like(C)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       # non-convergence is recorded explicitly
        sub, log = ot.sinkhorn(a[i], b[j], C[np.ix_(i, j)], eps, method="sinkhorn_log",
                               numItermax=MAX_ITER, stopThr=STOP_THR, log=True)
    P[np.ix_(i, j)] = sub
    return P, int(log["niter"]), float(log["err"][-1]) if len(log["err"]) else float("nan")


def run(task):
    out, path, method = task
    name = os.path.basename(path)[:-4]; z = np.load(path)
    C = cost_matrix(set_coords(z), set_coords(z), str(z["cost"]))
    plans, times, iters, errs = [], [], [], []
    for a, b in zip(z["a"], z["b"]):
        t0 = time.perf_counter()
        if method == "pot":
            P = ot.emd(a, b, C, numItermax=1_000_000)
        else:
            P, it, err = sinkhorn(a, b, C, float(method[len("sinkhorn_eps"):])); iters.append(it); errs.append(err)
        times.append(time.perf_counter() - t0); plans.append(P)
    info = {"method": method, "device": f"CPU ({platform.processor() or platform.machine()}), 1 thread",
            "single_instance_s_median": float(np.median(times)), "batched_s_per_instance": float(np.median(times))}
    if method != "pot":
        np.save(f"{out}/preds/{name}__{method}.npy", np.stack(plans))
        info.update({"stopThr": STOP_THR, "numItermax": MAX_ITER, "iterations_median": float(np.median(iters)),
                     "iterations_max": int(np.max(iters)), "converged_fraction": float(np.mean(np.array(errs) < STOP_THR))})
    json.dump(info, open(f"{out}/preds/{name}__{method}.timing.json", "w"), indent=2)
    return f"{method:<18} {name:<26} {np.median(times) * 1e3:9.3f} ms/instance" + (
        f"  converged {info['converged_fraction']:.1%}, median iterations {info['iterations_median']:.0f}" if method != "pot" else "")


if __name__ == "__main__":
    OUT = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 1)
    methods = ["pot"] + [f"sinkhorn_eps{e:g}" for e in EPS]
    tasks = [(OUT, p, m) for p in sorted(glob.glob(f"{OUT}/sets/*.npz")) for m in methods]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(run, tasks):
            print(line, flush=True)
