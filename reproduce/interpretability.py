"""Interpretability of the ToR-OT row model (App. D; env_torot, one GPU).

    python reproduce/interpretability.py experiments/torot/camera_ready experiments/torot/row_M_5M/best.pt \\
        [--set ood_paper_sqeuclidean]

The three studies of App. D on the instances of one evaluation set (default: the 144 OOD pairs, sqeuclidean). Plans are decoded row by row with the last row decoded by the model.

1. Causal intervention. For each instance, rows are re-decoded after changing only the remaining capacity r_j of
   the column j that receives the most mass in that row: exhausted (r_j = 0) or halved. The instance, the earlier
   rows and the weights are unchanged. Re-decoding without a change must reproduce the row (control).
2. Linear probes (ridge regression, held-out instances) on the decoder states at the position that predicts each
   mass (input token = the pointer to its column j), for every layer: the remaining row budget a_i - (mass already
   placed in the row) and the mass already placed (not inputs: they must be computed from the generated tokens),
   the remaining capacity r_j of the pointed column, and the whole capacity vector r. Controls: the embedding layer
   (the pointer's input embedding carries the pointed bin's encoder state, hence r_j) and a probe from r_j alone.
3. Attention heads. Attention weights of every decoder layer and head, recomputed from the layer's own q / k
   projections (checked against its output), averaged over the slots that predict masses: cross-attention to the
   cost token, the current source bin, the other source bins, the pointed target bin and the other target bins;
   self-attention to BOS, earlier pointers, earlier masses and the current entry.
Writes <root>/interp/{causal.json, causal.csv, probe.json, attention.npz}; make_figures.py draws App. D figures.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch

from torot.data.cost import COST_TYPES
from torot.data.dataset import Collator, row_subproblem
from torot.data.supports import grid_1d, pad_coords
from torot.decode import _to_device, decode_by_rows, greedy_tokens
from torot.evaluate import load_model

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("checkpoint")
ap.add_argument("--set", default="ood_paper_sqeuclidean")
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None, help="output folder (default: <root>/interp)")
args = ap.parse_args()
OUT = args.out or f"{args.root}/interp"
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")
model, tok, cfg = load_model(args.checkpoint, dev)
model.eval()
collate = Collator(tok)

z = np.load(f"{args.root}/sets/{args.set}.npz")
n = int(z["n"]); coords = pad_coords(grid_1d(n))
insts = [{"n": n, "m": n, "dim": 1, "cost_type": COST_TYPES.index(str(z["cost"])), "coords_src": coords,
          "coords_tgt": coords, "a": a, "b": b} for a, b in zip(z["a"], z["b"])]

# --------------------------------------------------------------------------------------------- decode with a trace
trace: list = []
with torch.no_grad():
    decode_by_rows(model, tok, insts, dev, last_row="model", trace=trace)
for t in trace:
    t["cols"], t["vals"] = tok.decode_row(t["ids"])
print(f"{len(trace)} rows decoded", flush=True)


def decode_rows(subs):
    out = []
    for s in range(0, len(subs), args.batch):
        batch = _to_device(collate(subs[s:s + args.batch]), dev)
        out += greedy_tokens(model, tok, batch, (tok.D + 2) * n + 2, single_row=True)
    return [tok.decode_row(ids) for ids in out]


# --------------------------------------------------------------------------------------------- 1. causal intervention
rng = np.random.default_rng(args.seed)
picked = []
for k in range(len(insts)):
    rows = [t for t in trace if t["instance"] == k and t["row"] < n - 1 and len(t["cols"]) > 0]
    for t in rng.permutation(len(rows))[:2]:                       # two rows per instance
        picked.append(rows[t])
subs, meta = [], []
for t in picked:
    j = int(t["cols"][np.argmax(t["vals"])])
    for kind, factor in (("control", 1.0), ("exhaust", 0.0), ("halve", 0.5)):
        cap = t["capacity"].copy(); cap[j] *= factor
        subs.append(row_subproblem(insts[t["instance"]], t["row"], cap)); meta.append((t, j, kind, cap[j]))
with torch.no_grad():
    res = decode_rows(subs)
recs = []
for (t, j, kind, cap_j), (cols, vals) in zip(meta, res):
    old = dict(zip(t["cols"].tolist(), t["vals"].tolist())); new = dict(zip(cols.tolist(), vals.tolist()))
    m_old, m_new = old.get(j, 0.0), new.get(j, 0.0)
    removed = max(m_old - m_new, 0.0)
    moved_to_neighbours = sum(max(new.get(c, 0.0) - old.get(c, 0.0), 0.0) for c in (j - 1, j + 1))
    recs.append({"instance": t["instance"], "row": t["row"], "column": j, "intervention": kind, "capacity_j": cap_j,
                 "mass_before": m_old, "mass_after": m_new, "reduction": 1 - m_new / m_old if m_old > 0 else np.nan,
                 "row_mass_before": sum(old.values()), "row_mass_after": sum(new.values()),
                 "respects_capacity": m_new <= cap_j + 1e-4, "removed_mass_to_neighbours": moved_to_neighbours / removed if removed > 1e-9 else np.nan,
                 "unchanged_row": old == new})
df = pd.DataFrame(recs); df.to_csv(f"{OUT}/causal.csv", index=False)
summ = {"set": args.set, "interventions_per_kind": int((df.intervention == "exhaust").sum())}
for kind in ("control", "exhaust", "halve"):
    d = df[df.intervention == kind]
    summ[kind] = {"unchanged_row": float(d.unchanged_row.mean()), "adapted (mass to j reduced)": float((d.reduction > 1e-6).mean()),
                  "mean_reduction": float(d.reduction.mean()), "complete_avoidance": float((d.mass_after == 0).mean()),
                  "respects_new_capacity": float(d.respects_capacity.mean()),
                  "row_mass_change_mean": float((d.row_mass_after - d.row_mass_before).mean()),
                  "share_of_removed_mass_to_adjacent_columns": float(d.removed_mass_to_neighbours.mean())}
json.dump(summ, open(f"{OUT}/causal.json", "w"), indent=2)
print(json.dumps(summ, indent=1), flush=True)

# --------------------------------------------------------------------------------------------- 2-3. teacher-forced pass
L, H = len(model.dec_layers), model.dec_layers[0].self_attn.n_heads
hidden, attn_in = {}, {}


def keep_hidden(name):
    def hook(module, inp, out):
        hidden[name] = out.detach()
    return hook


def keep_attn(name):
    def hook(module, a, kw, out):
        attn_in[name] = (module, a[0].detach(), kw.get("ctx"), kw.get("key_mask"), kw.get("rope"), out.detach())
    return hook


handles = [model.dec_norm.register_forward_hook(keep_hidden("final"))]
for l, layer in enumerate(model.dec_layers):
    handles.append(layer.register_forward_hook(keep_hidden(f"layer{l}")))
    handles.append(layer.self_attn.register_forward_hook(keep_attn(f"self{l}"), with_kwargs=True))
    handles.append(layer.cross_attn.register_forward_hook(keep_attn(f"cross{l}"), with_kwargs=True))


def weights(module, x, ctx, key_mask, rope):
    """Attention probabilities [B, H, T, S] of an Attention module, recomputed like its forward pass."""
    q = module.q_norm(module._split(module.wq(x)))
    src = x if ctx is None else ctx
    k = module.k_norm(module._split(module.wk(src)))
    if ctx is None and rope is not None:
        q, k = rope(q, 0), rope(k, 0)
    s = q @ k.transpose(-1, -2) / math.sqrt(module.head_dim)
    if key_mask is not None:
        s = s.masked_fill(~key_mask[:, None, None, :], float("-inf"))
    if ctx is None:
        T = s.shape[-1]
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=s.device), 1), float("-inf"))
    return s.softmax(-1), module._split(module.wv(src))


feats = {f"layer{l}": [] for l in range(L)} | {"embedding": []}
targets = {"row_budget": [], "placed_in_row": [], "capacity_j": [], "capacity_vector": [], "entry": [], "instance": []}
CROSS = ["cost token", "current source bin", "other source bins", "pointed target bin", "other target bins"]
SELF = ["BOS", "earlier pointers", "earlier masses", "current entry"]
cross_acc, self_acc, slots = np.zeros((L, H, len(CROSS))), np.zeros((L, H, len(SELF))), 0
checked, recheck = False, {}                                      # relative error of the recomputed attention outputs
for s0 in range(0, len(trace), args.batch):
    rows = trace[s0:s0 + args.batch]
    batch = _to_device(collate([row_subproblem(insts[t["instance"]], t["row"], t["capacity"]) for t in rows]), dev)
    T = max(len(t["ids"]) for t in rows) + 1
    ids = torch.full((len(rows), T), tok.EOS, dtype=torch.long, device=dev)
    for r, t in enumerate(rows):
        ids[r, : len(t["ids"]) + 1] = torch.tensor([tok.BOS] + t["ids"], device=dev)
    with torch.no_grad():
        enc = model.encode(batch)
        emb = model.embed_tokens(ids, enc)
        model.decode(ids, enc)
        A = {}
        for l in range(L):
            for kind in ("self", "cross"):
                module, x, ctx, km, rope, out = attn_in[f"{kind}{l}"]
                w, v = weights(module, x, ctx, km, rope)
                if not checked:                                   # the recomputed weights reproduce the layer
                    rec = module.wo((w @ v).transpose(1, 2).reshape(out.shape))
                    recheck[f"{kind}{l}"] = float((rec - out).norm() / out.norm())
                A[kind, l] = w.float().cpu().numpy()
            checked = True
    n_pad = batch["a"].shape[1]
    for r, t in enumerate(rows):
        seq = [tok.BOS] + t["ids"]
        a_i, cap = float(insts[t["instance"]]["a"][t["row"]]), t["capacity"]
        placed, entry, entry_start = 0.0, -1, None
        mass_slots = []                                           # (position, pointed column, entry start)
        for p, tid in enumerate(seq):
            if tok.is_pointer(tid):
                entry += 1; entry_start = p
                j = tid - tok.vocab_size
                for name in feats:
                    src = emb if name == "embedding" else hidden[name]
                    feats[name].append(src[r, p].float().cpu().numpy())
                targets["row_budget"].append(a_i - placed); targets["placed_in_row"].append(placed)
                targets["capacity_j"].append(float(cap[j]))
                targets["capacity_vector"].append(cap.copy()); targets["entry"].append(entry)
                targets["instance"].append(t["instance"])
                placed += float(t["vals"][entry]) if entry < len(t["vals"]) else 0.0
                mass_slots.append((p, j, p))
            elif (tok.is_exponent(tid) or tok.is_digit(tid)) and entry_start is not None:
                mass_slots.append((p, seq[entry_start] - tok.vocab_size, entry_start))
        for p, j, start in mass_slots:
            slots += 1
            for l in range(L):
                c = A["cross", l][r, :, p]                       # [H, S]: 0 cost, 1..n_pad source (1 = current), then targets
                cross_acc[l] += np.stack([c[:, 0], c[:, 1], c[:, 2:1 + n_pad].sum(1), c[:, 1 + n_pad + j],
                                          c[:, 1 + n_pad:].sum(1) - c[:, 1 + n_pad + j]], 1)
                sa = A["self", l][r, :, p, : p + 1]              # [H, p+1]
                kinds = np.array(["BOS"] + ["ptr" if tok.is_pointer(x) else "mass" for x in seq[1:p + 1]])
                cur = np.arange(p + 1) >= start
                self_acc[l] += np.stack([sa[:, 0], sa[:, (kinds == "ptr") & ~cur].sum(1), sa[:, (kinds == "mass") & ~cur].sum(1),
                                         sa[:, cur].sum(1)], 1)
for h in handles:
    h.remove()
np.savez(f"{OUT}/attention.npz", cross=cross_acc / slots, self=self_acc / slots, cross_labels=CROSS, self_labels=SELF, slots=slots)
json.dump({"slots": slots, "max_relative_error_of_recomputed_attention": max(recheck.values()), "per_module": recheck},
          open(f"{OUT}/attention.json", "w"), indent=1)
print("attention recomputation, max relative error:", round(max(recheck.values()), 5), flush=True)


def ridge_r2(X, Y, train, test, lam=1.0):
    """Ridge regression on standardised features; R^2 on the test rows (averaged over output columns)."""
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    Xs = (X - mu) / sd
    Xt = np.c_[Xs, np.ones(len(Xs))]
    ym = Y[train].mean(0)
    W = np.linalg.solve(Xt[train].T @ Xt[train] + lam * np.eye(Xt.shape[1]), Xt[train].T @ (Y[train] - ym))
    pred = Xt[test] @ W + ym
    ss_res, ss_tot = ((Y[test] - pred) ** 2).sum(0), ((Y[test] - Y[test].mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-12)))


inst = np.array(targets["instance"]); entry = np.array(targets["entry"])
test = np.isin(inst, rng.permutation(len(insts))[: len(insts) // 4]); train = ~test     # hold out a quarter of instances
probe = {"set": args.set, "positions": int(len(inst)), "held_out_instances": int(len(np.unique(inst[test]))), "by_layer": {}}
Ys = {"row_budget": np.array(targets["row_budget"])[:, None], "placed_in_row": np.array(targets["placed_in_row"])[:, None],
      "capacity_j": np.array(targets["capacity_j"])[:, None], "capacity_vector": np.array(targets["capacity_vector"])}
for name in ["embedding"] + [f"layer{l}" for l in range(L)]:
    X = np.stack(feats[name])
    probe["by_layer"][name] = {t: ridge_r2(X, Y, train, test) for t, Y in Ys.items()}
# controls: how much of each scalar target the pointed column's capacity alone explains (it is in the input embedding)
cj = np.array(targets["capacity_j"])[:, None]
probe["control_from_capacity_j_only"] = {t: ridge_r2(np.c_[cj, np.log(cj + 1e-9)], Y, train, test)
                                         for t, Y in Ys.items() if t in ("row_budget", "placed_in_row")}
X = np.stack(feats[f"layer{L - 1}"])
probe["row_budget_by_entry_last_layer"] = {int(e): ridge_r2(X, Ys["row_budget"], train, test & (entry == e))
                                            for e in range(int(entry.max()) + 1) if (test & (entry == e)).sum() >= 20}
json.dump(probe, open(f"{OUT}/probe.json", "w"), indent=2)
print(json.dumps({k: v for k, v in probe["by_layer"].items() if k in ("embedding", f"layer{L - 1}")}, indent=1), flush=True)
