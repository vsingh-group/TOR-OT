"""Grammar-constrained greedy decoding with a KV cache, and transpose-averaged prediction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from torot.data.dataset import Collator, transpose_instance
from torot.model import ToROT
from torot.tokenizer import GrammarState, PlanTokenizer

Plan = Tuple[np.ndarray, np.ndarray, np.ndarray]  # rows, cols, vals


@torch.no_grad()
def generate(
    model: ToROT,
    tok: PlanTokenizer,
    batch: Dict[str, Any],
    max_len: int = 2048,
) -> List[Plan]:
    """Greedy decoding constrained by the serialisation grammar. Returns one COO plan per instance."""
    model.eval()
    enc = model.encode(batch)
    device = batch["a"].device
    B = batch["a"].shape[0]
    m_pad = batch["b"].shape[1]
    n_list, m_list = batch["n"].tolist(), batch["m"].tolist()

    states = [GrammarState() for _ in range(B)]
    cache = model.new_cache()
    cur = torch.full((B, 1), tok.BOS, dtype=torch.long, device=device)
    generated: List[List[int]] = [[] for _ in range(B)]

    for step in range(max_len):
        logits = model.decode(cur, enc, cache=cache, offset=step)[:, -1].float()
        mask = np.stack([tok.allowed_mask(s, n_list[i], m_list[i], m_pad) for i, s in enumerate(states)])
        logits = logits.masked_fill(~torch.from_numpy(mask).to(device), float("-inf"))
        nxt = logits.argmax(-1)
        nxt_list = nxt.tolist()
        for i, t in enumerate(nxt_list):
            if not states[i].finished:
                generated[i].append(t)
                tok.advance(states[i], t)
        if all(s.finished for s in states):
            break
        cur = nxt.unsqueeze(1)

    return [tok.decode(g) for g in generated]


def plan_to_dense(plan: Plan, n: int, m: int) -> np.ndarray:
    rows, cols, vals = plan
    P = np.zeros((n, m), dtype=np.float64)
    np.add.at(P, (rows, cols), vals)
    return P


@torch.no_grad()
def predict_plans(
    model: ToROT,
    tok: PlanTokenizer,
    instances: List[Dict[str, Any]],
    device: torch.device,
    max_len: int = 2048,
    symmetric: bool = False,
) -> List[np.ndarray]:
    """Dense predicted plans for a list of instances.

    With ``symmetric=True`` (symmetry 6.1 at test time) the instance is also decoded with
    source and target swapped and the transposed plan is averaged with the direct one.
    """
    collate = Collator(tok)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in collate(instances).items()}
    plans = [plan_to_dense(p, inst["n"], inst["m"]) for p, inst in zip(generate(model, tok, batch, max_len), instances)]
    if not symmetric:
        return plans
    swapped = [transpose_instance(inst) for inst in instances]
    batch_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in collate(swapped).items()}
    plans_t = [plan_to_dense(p, inst["n"], inst["m"]).T for p, inst in zip(generate(model, tok, batch_t, max_len), swapped)]
    return [0.5 * (p + q) for p, q in zip(plans, plans_t)]
