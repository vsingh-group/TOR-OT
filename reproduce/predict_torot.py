"""Predictions of the ToR-OT row-mode model on the evaluation sets (env_torot).

    python reproduce/predict_torot.py experiments/torot/camera_ready experiments/torot/row_M_5M/best.pt [set patterns...]   (env_torot)

Optional set patterns (fnmatch on the set name, e.g. 'ood_paper_*') restrict which sets are predicted. The checkpoint
can also be an exported folder or a Hub id (ivanjaenm/torot-1d), see torot.evaluate.load_model.
Greedy, grammar-constrained row decoding with every row decoded by the model (the paper's decoding,
torot.decode.decode_by_rows(last_row="model")); --last-row remainder instead sets the last row to the remaining
capacity (outputs tagged without _lastrow). --plain-only skips the transpose-averaged pass. --device cpu decodes on CPU
(same greedy decoding, same plans) and writes no timing file, since CPU times are not comparable with the GPU timings
of the other methods.

Writes preds/<set>__v2row_lastrow.npy and preds/<set>__v2row_sym_lastrow.npy (transpose-averaged), plus *.timing.json.
"""
import fnmatch, glob, json, os, sys, time
import numpy as np, torch
from torot.data.cost import COST_TYPES
from torot.data.supports import pad_coords, set_coords
from torot.decode import predict_plans
from torot.evaluate import load_model

args = sys.argv[1:]
LAST_ROW = "model"                           # the paper's decoding; --last-row remainder: the last row = remaining capacity
if "--last-row" in args:
    i = args.index("--last-row"); LAST_ROW = args[i + 1]; del args[i:i + 2]
OUT, CKPT = args[0], args[1]
PATTERNS = args[2:] or ["*"]
SUFFIX = "" if LAST_ROW == "remainder" else "_lastrow"
PLAIN_ONLY = "--plain-only" in args          # skip the transpose-averaged pass
if PLAIN_ONLY:
    args.remove("--plain-only")
DEVICE = "cuda"
if "--device" in args:
    i = args.index("--device"); DEVICE = args[i + 1]; del args[i:i + 2]
OUT, CKPT = args[0], args[1]                 # re-read: the options above are removed from args
PATTERNS = args[2:] or ["*"]
dev = torch.device(DEVICE)
CUDA = dev.type == "cuda"
model, tok, cfg = load_model(CKPT, dev)

def instances(z):
    n = int(z["n"]); coords = pad_coords(set_coords(z)); dim = int(z["dim"]) if "dim" in z else 1; res = []
    for a, b, P, c in zip(z["a"], z["b"], z["plan"], z["ot_cost"]):
        r, k = np.nonzero(P)
        res.append({"n": n, "m": n, "dim": dim, "cost_type": COST_TYPES.index(str(z["cost"])), "coords_src": coords,
                    "coords_tgt": coords, "a": a, "b": b, "plan_rows": r, "plan_cols": k, "plan_vals": P[r, k], "ot_cost": float(c)})
    return res

def timed(fn):
    if CUDA: torch.cuda.synchronize()
    t0 = time.perf_counter(); out = fn()
    if CUDA: torch.cuda.synchronize()
    return out, time.perf_counter() - t0

os.makedirs(f"{OUT}/preds", exist_ok=True)
for path in sorted(glob.glob(f"{OUT}/sets/*.npz")):
    name = os.path.basename(path)[:-4]
    if not any(fnmatch.fnmatch(name, p) for p in PATTERNS):
        continue
    insts = instances(np.load(path))
    predict_plans(model, tok, insts[:1], dev)                                        # warm-up
    for sym, tag in ((False, "v2row" + SUFFIX),) + (() if PLAIN_ONLY else ((True, "v2row_sym" + SUFFIX),)):
        plans, total = [], 0.0
        for s in range(0, len(insts), 128):
            P, dt = timed(lambda: predict_plans(model, tok, insts[s:s + 128], dev, symmetric=sym, last_row=LAST_ROW)); plans += P; total += dt
        np.save(f"{OUT}/preds/{name}__{tag}.npy", np.stack(plans))
        if not CUDA:
            print(f"{tag} {name}: {len(plans)} plans ({DEVICE}, no timing file)", flush=True); continue
        single = [timed(lambda k=k: predict_plans(model, tok, insts[k:k + 1], dev, symmetric=sym, last_row=LAST_ROW))[1] for k in range(10)]
        json.dump({"method": tag, "checkpoint": CKPT, "last_row": LAST_ROW, "device": torch.cuda.get_device_name(),
                   "batched_s_per_instance": total / len(insts), "batch_instances": 128,
                   "single_instance_s_median": float(np.median(single))}, open(f"{OUT}/preds/{name}__{tag}.timing.json", "w"), indent=2)
        print(f"{tag} {name}: {total / len(insts) * 1e3:.1f} ms/instance batched, {np.median(single):.2f} s single", flush=True)
