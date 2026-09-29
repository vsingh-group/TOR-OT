"""
Rebuttal experiment (W1/Q1 scaling behavior):
One ToR-OT checkpoint (trained at max support 30), evaluated at effective
support n = 10/15/20/25/30 by zero-padding the marginals to the trained maximum.

Reports per effective support size:
  - mean sequence length (generated tokens)
  - inference latency per instance
  - MAPE
  - OT cost gap
  - max marginal violation

POT (ot.solve) is used ONLY to build the ground-truth LP plans that the
metrics are computed against; it is not a compared method here.

Run:
  PYTHONPATH=. /home/ivan/anaconda3/envs/env_TF4OT/bin/python TF4OT/rebuttal_variable_support.py
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import ot
from scipy import sparse
from tabulate import tabulate

from TF4OT import utils
from TF4OT.utils import my_dtype_np, my_dtype_torch
from TF4OT.run_inference import load_model_by_name

MODEL_DIR = ("experiments/cost_fixed/"
             "model_dim768_context512_tflayers24_tfheads16_num_prefix1_dropout0_"
             "flashTrue_data100K_rowwise_costfixed_prec4_bins30_DigitTokenizer_"
             "epochs50_batch192_sqeuclidean")

N_MAX          = 30                       # trained maximum support
SUPPORT_SIZES  = [10, 15, 20, 25, 30]     # effective supports to simulate
NUM_INSTANCES  = int(os.environ.get("NUM_INSTANCES", 100))  # instances per support size
MAX_SEQ_LEN    = 64                       # per-row token budget (as in inference cfg)
TEMPERATURE    = 0.1                      # as in eval_all_testset
PRECISION      = 4                        # DigitTokenizer decimal precision
SEED           = 0


@torch.no_grad()
def generate_early_stop(model, tokenizer, encoder_input_ab, max_seq_len, temperature):
    """
    Same autoregressive loop as TokenRegressor_flash.generate, but stops as soon
    as every row in the batch has emitted the ']' token, so wall time reflects
    the actual generated sequence length.
    Returns (token_ids [B, steps], seq_lens [B]) where seq_len counts tokens up
    to and including the first ']' (max_seq_len if never closed).
    """
    device     = encoder_input_ab.device
    batch_size = encoder_input_ab.shape[0]
    end_id     = tokenizer.vocabulary_mapping[']']

    token_ids = torch.full((batch_size, max_seq_len), -1, dtype=torch.long, device=device)
    finished  = torch.zeros(batch_size, dtype=torch.bool, device=device)
    steps     = 0

    for i in range(max_seq_len):
        logits = model.forward(encoder_input_ab, token_ids[:, :i], encoder_input_c=None, pad_mask=None)
        probs  = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        token_ids[:, i] = next_token
        finished |= (next_token == end_id)
        steps = i + 1
        if bool(finished.all()):
            break

    # per-row sequence length = first ']' position + 1
    ids_np   = token_ids[:, :steps].cpu().numpy()
    seq_lens = np.full(batch_size, max_seq_len, dtype=np.int64)
    for b in range(batch_size):
        hits = np.where(ids_np[b] == end_id)[0]
        if hits.size > 0:
            seq_lens[b] = hits[0] + 1
    return token_ids[:, :steps], seq_lens


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    from omegaconf import OmegaConf
    cfg_loaded = OmegaConf.load(os.path.join(MODEL_DIR, "config.yaml"))
    if "logging" in cfg_loaded:  # hydra interpolations in logging break resolve
        del cfg_loaded["logging"]
    cfg = OmegaConf.to_container(cfg_loaded, resolve=True)
    model, tokenizer = load_model_by_name(MODEL_DIR, device=device, cfg=cfg)
    model.eval()

    # Fixed cost matrix exactly as in training data generation (run_datagenerator)
    xs = np.arange(N_MAX, dtype=my_dtype_np)[:, None]
    C = ot.dist(xs, xs, metric="sqeuclidean")
    C = C / C.max()

    results = {}
    for n in SUPPORT_SIZES:
        inst = {k: [] for k in
                ["latency", "tokens_total", "tokens_effective", "tokens_padded",
                 "gt_nnz", "mape", "otcost_gap", "otcost_rel_gap", "max_violation",
                 "unclosed_rows"]}

        for t in range(NUM_INSTANCES):
            # marginals supported on the first n bins, zero-padded to N_MAX
            a = np.zeros(N_MAX, dtype=np.float64)
            b = np.zeros(N_MAX, dtype=np.float64)
            a[:n] = rng.uniform(size=n); a /= a.sum()
            b[:n] = rng.uniform(size=n); b /= b.sum()

            # ground-truth LP plan (reference only, not a baseline)
            gt_plan = ot.solve(C, a=a, b=b).plan
            gt_sparse = sparse.coo_matrix(gt_plan)
            otcost_gt = float((gt_plan * C).sum())

            # encoder inputs: one row-conditional per source bin with mass,
            # [a | b | j/N_MAX]. Rows with a_j = 0 are fixed to zero a priori
            # (their plan row is empty by feasibility), so only n rows are decoded.
            ab = np.concatenate([a, b])
            enc = np.stack([np.concatenate([ab, [j / N_MAX]]) for j in range(n)])
            enc = torch.from_numpy(enc).to(dtype=my_dtype_torch, device=device)

            if t == 0:  # warmup (not timed)
                generate_early_stop(model, tokenizer, enc, MAX_SEQ_LEN, TEMPERATURE)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            token_ids, seq_lens = generate_early_stop(model, tokenizer, enc, MAX_SEQ_LEN, TEMPERATURE)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency = time.perf_counter() - t0

            # decode rows -> dense predicted plan
            texts = tokenizer.from_tokens(token_ids)
            texts = [s[:s.find(']') + 1] if (']' in s) else s for s in texts]
            rows = [utils.parse_predicted_sparse_row_string(s, N_MAX, mode="index", precision=PRECISION)
                    for s in texts]
            pred_plan = np.zeros((N_MAX, N_MAX))
            pred_plan[:n] = np.stack([np.fromstring(r[1:-1], sep=',') for r in rows])
            pred_sparse = sparse.coo_matrix(pred_plan)

            viol = utils.compute_constraint_violations(pred_sparse, a, b)
            otcost_pred = float((pred_plan * C).sum())

            inst["latency"].append(latency)
            inst["tokens_total"].append(int(seq_lens.sum()))
            inst["tokens_effective"].append(int(seq_lens.sum()))
            inst["tokens_padded"].append(0)
            inst["gt_nnz"].append(int(gt_sparse.nnz))
            inst["mape"].append(utils.sparse_weighted_mean_abs_percentage_error(gt_sparse, pred_sparse))
            inst["otcost_gap"].append(abs(otcost_pred - otcost_gt))
            inst["otcost_rel_gap"].append(abs(otcost_pred - otcost_gt) / max(otcost_gt, 1e-12))
            inst["max_violation"].append(viol["max_violation"])
            inst["unclosed_rows"].append(int((seq_lens >= MAX_SEQ_LEN).sum()))

        results[n] = {k: np.array(v) for k, v in inst.items()}
        r = results[n]
        print(f"[n={n:2d}] gt_nnz={r['gt_nnz'].mean():6.1f}  "
              f"tok_eff={r['tokens_effective'].mean():6.1f}  tok_tot={r['tokens_total'].mean():6.1f}  "
              f"lat={r['latency'].mean()*1000:7.1f}ms  mape={r['mape'].mean():.4f}  "
              f"gap={r['otcost_gap'].mean():.2e} (rel {r['otcost_rel_gap'].mean():.2e})  "
              f"maxviol={r['max_violation'].mean():.2e}", flush=True)

    # save raw results
    np.savez(os.path.join(MODEL_DIR, "rebuttal_variable_support_results.npz"),
             **{f"n{n}_{k}": v for n, r in results.items() for k, v in r.items()})

    # final table
    headers = ["Effective support n", "GT nnz (mean)", "Mean seq. len (tokens)",
               "Latency (ms)", "MAPE", "OT cost gap (abs)", "OT cost gap (rel)",
               "Max marginal violation"]
    table = []
    for n in SUPPORT_SIZES:
        r = results[n]
        table.append([
            n,
            f"{r['gt_nnz'].mean():.1f}",
            f"{r['tokens_effective'].mean():.1f} ± {r['tokens_effective'].std():.1f}",
            f"{r['latency'].mean()*1000:.1f} ± {r['latency'].std()*1000:.1f}",
            f"{r['mape'].mean():.4f}",
            f"{r['otcost_gap'].mean():.2e}",
            f"{r['otcost_rel_gap'].mean():.2e}",
            f"{r['max_violation'].mean():.2e}",
        ])
    print()
    print(tabulate(table, headers=headers, tablefmt="github"))
    print(f"\n(NUM_INSTANCES={NUM_INSTANCES} per support size; only the n rows with nonzero "
          f"source mass are decoded — rows with a_j=0 are zero by feasibility)")


if __name__ == "__main__":
    main()
