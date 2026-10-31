"""Grammar-constrained greedy decoding with a KV cache, and transpose-averaged prediction.

``predict_plans`` dispatches on ``model.cfg.decoding``:

* ``plan``: one sequence per instance, decoded in a single pass.
* ``row``: rows are decoded one at a time. Row i is generated from the remaining problem
  (source bins i..n-1, remaining column capacities) and its masses are subtracted from the
  capacities. The last row is either set to whatever capacity remains (``last_row="remainder"``,
  the default: column marginals exact up to over-filled columns, but every small error of the
  earlier rows ends up there as a scattering of tiny entries) or decoded by the model like the
  other rows (``last_row="model"``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from torot.data.dataset import Collator, row_subproblem, transpose_instance
from torot.model import ToROT
from torot.tokenizer import GrammarState, PlanTokenizer

Plan = Tuple[np.ndarray, np.ndarray, np.ndarray]  # rows, cols, vals


@torch.no_grad()
def greedy_tokens(
    model: ToROT,
    tok: PlanTokenizer,
    batch: Dict[str, Any],
    max_len: int,
    single_row: bool = False,
) -> List[List[int]]:
    """Greedy decoding constrained by the serialisation grammar; returns the generated ids
    (without BOS) for every sequence of the batch."""
    model.eval()
    enc = model.encode(batch)
    device = batch["a"].device
    B = batch["a"].shape[0]
    m_pad = batch["b"].shape[1]
    n_list = [1] * B if single_row else batch["n"].tolist()
    m_list = batch["m"].tolist()

    states = [tok.row_state() if single_row else GrammarState() for _ in range(B)]
    cache = model.new_cache()
    cur = torch.full((B, 1), tok.BOS, dtype=torch.long, device=device)
    generated: List[List[int]] = [[] for _ in range(B)]

    for step in range(max_len):
        logits = model.decode(cur, enc, cache=cache, offset=step)[:, -1].float()
        mask = np.stack([tok.allowed_mask(s, n_list[i], m_list[i], m_pad) for i, s in enumerate(states)])
        logits = logits.masked_fill(~torch.from_numpy(mask).to(device), float("-inf"))
        nxt = logits.argmax(-1)
        for i, t in enumerate(nxt.tolist()):
            if not states[i].finished:
                generated[i].append(t)
                tok.advance(states[i], t)
        if all(s.finished for s in states):
            break
        cur = nxt.unsqueeze(1)
    return generated


def generate(model: ToROT, tok: PlanTokenizer, batch: Dict[str, Any], max_len: int = 2048) -> List[Plan]:
    """Whole-plan decoding (``decoding: plan``). Returns one COO plan per instance."""
    return [tok.decode(g) for g in greedy_tokens(model, tok, batch, max_len)]


def plan_to_dense(plan: Plan, n: int, m: int) -> np.ndarray:
    rows, cols, vals = plan
    P = np.zeros((n, m), dtype=np.float64)
    np.add.at(P, (rows, cols), vals)
    return P


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def decode_by_rows(
    model: ToROT,
    tok: PlanTokenizer,
    instances: List[Dict[str, Any]],
    device: torch.device,
    max_len: int = 2048,
    last_row: str = "remainder",
    trace: List[Dict[str, Any]] | None = None,
) -> List[np.ndarray]:
    """Row decoding (``decoding: row``): dense plans built one row at a time.

    With ``trace`` (a list), every decoded row is appended as {"instance", "row", "capacity" (before the row),
    "ids" (generated tokens)}, e.g. for interpretability analyses.
    """
    if last_row not in ("remainder", "model"):
        raise ValueError(f"last_row must be 'remainder' or 'model', got {last_row!r}")
    n_rest = 0 if last_row == "model" else 1       # trailing rows filled by the remainder rule
    collate = Collator(tok)
    plans = [np.zeros((inst["n"], inst["m"]), dtype=np.float64) for inst in instances]
    capacity = [np.asarray(inst["b"], dtype=np.float64).copy() for inst in instances]
    row_len = min(max_len, (tok.D + 2) * max(inst["m"] for inst in instances) + 2)   # at most m entries per row
    for i in range(max(inst["n"] for inst in instances) - n_rest):
        active = [k for k, inst in enumerate(instances) if i < inst["n"] - n_rest]
        if not active:
            break
        batch = _to_device(collate([row_subproblem(instances[k], i, capacity[k]) for k in active]), device)
        for k, ids in zip(active, greedy_tokens(model, tok, batch, row_len, single_row=True)):
            if trace is not None:
                trace.append({"instance": k, "row": i, "capacity": capacity[k].copy(), "ids": list(ids)})
            cols, vals = tok.decode_row(ids)
            np.add.at(plans[k][i], cols, vals)
            capacity[k] = np.maximum(capacity[k] - plans[k][i], 0.0)
    if last_row == "remainder":
        for k, inst in enumerate(instances):
            plans[k][inst["n"] - 1] = capacity[k]      # the last row takes what remains
    return plans


@torch.no_grad()
def _predict(model: ToROT, tok: PlanTokenizer, instances: List[Dict[str, Any]], device: torch.device, max_len: int,
             last_row: str = "remainder") -> List[np.ndarray]:
    if model.cfg.decoding == "row":
        return decode_by_rows(model, tok, instances, device, max_len, last_row)
    batch = _to_device(Collator(tok)(instances), device)
    return [plan_to_dense(p, inst["n"], inst["m"]) for p, inst in zip(generate(model, tok, batch, max_len), instances)]


@torch.no_grad()
def predict_plans(
    model: ToROT,
    tok: PlanTokenizer,
    instances: List[Dict[str, Any]],
    device: torch.device,
    max_len: int = 2048,
    symmetric: bool = False,
    last_row: str = "remainder",
) -> List[np.ndarray]:
    """Dense predicted plans for a list of instances (either decoding mode).

    With ``symmetric=True`` (symmetry 6.1 at test time) the instance is also decoded with
    source and target swapped and the transposed plan is averaged with the direct one.
    ``last_row`` applies to row decoding (see ``decode_by_rows``).
    """
    plans = _predict(model, tok, instances, device, max_len, last_row)
    if not symmetric:
        return plans
    swapped = [transpose_instance(inst) for inst in instances]
    plans_t = [P.T for P in _predict(model, tok, swapped, device, max_len, last_row)]
    return [0.5 * (p + q) for p, q in zip(plans, plans_t)]
