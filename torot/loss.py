"""Significance-weighted, ordinal cross-entropy over the extended vocabulary.

Two departures from plain token cross-entropy, both motivated by the fact that the
targets are numbers rather than words:

* **Significance weights.** The exponent and the first mantissa digit carry most of the
  numeric error; the p-th digit is weighted ``decay ** (p - 1)``. Structural and pointer
  slots keep weight 1.
* **Ordinal soft labels.** For digit and exponent slots a small probability ``eps`` is
  moved to each numeric neighbour (d-1, d+1), so predicting 3 for a true 4 costs less
  than predicting 9. No auxiliary numeric loss is used.
* **z-loss.** ``z_loss * logsumexp(logits)^2`` keeps the logits from drifting to large
  magnitudes (a known source of training instability); it does not change the optimum.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torot.tokenizer import KIND_DIGIT, KIND_EXP, KIND_PAD, KIND_STRUCT, KIND_TGT, PlanTokenizer


class PlanLoss(nn.Module):
    def __init__(
        self,
        tokenizer: PlanTokenizer,
        digit_weight_decay: float = 0.5,
        exponent_weight: float = 1.0,
        ordinal_eps: float = 0.05,
        z_loss: float = 0.0,
    ):
        super().__init__()
        self.tok = tokenizer
        self.eps = float(ordinal_eps)
        self.z_loss = float(z_loss)
        weights = torch.ones(KIND_DIGIT + tokenizer.D)
        weights[KIND_EXP] = exponent_weight
        for p in range(tokenizer.D):
            weights[KIND_DIGIT + p] = digit_weight_decay**p
        self.register_buffer("kind_weights", weights, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, kinds: torch.Tensor) -> torch.Tensor:
        """logits [B, T, E], targets [B, T] (ids), kinds [B, T] (KIND_PAD for padding)."""
        tok = self.tok
        E = logits.shape[-1]
        logits = logits.float()
        logp = F.log_softmax(logits, dim=-1)
        valid = kinds != KIND_PAD
        kinds_c = kinds.clamp(min=0)
        tgt = targets.clamp(min=0, max=E - 1)

        lp_t = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

        is_digit = kinds_c >= KIND_DIGIT
        is_exp = kinds_c == KIND_EXP
        lo, hi = tgt - 1, tgt + 1
        lo_ok = (is_digit & (lo >= tok.digit0)) | (is_exp & (lo >= tok.exp0))
        hi_ok = (is_digit & (hi < tok.digit0 + 10)) | (is_exp & (hi < tok.exp0 + tok.max_exponent))
        lp_lo = logp.gather(-1, lo.clamp(min=0, max=E - 1).unsqueeze(-1)).squeeze(-1)
        lp_hi = logp.gather(-1, hi.clamp(min=0, max=E - 1).unsqueeze(-1)).squeeze(-1)
        # neighbours of non-numeric slots may be masked (-inf) pointer logits: select, never multiply by 0
        zero = torch.zeros_like(lp_t)
        lp_lo = torch.where(lo_ok, lp_lo, zero)
        lp_hi = torch.where(hi_ok, lp_hi, zero)
        eps_lo = self.eps * lo_ok.float()
        eps_hi = self.eps * hi_ok.float()
        nll = -((1.0 - eps_lo - eps_hi) * lp_t + eps_lo * lp_lo + eps_hi * lp_hi)

        w = self.kind_weights[kinds_c] * valid.float()
        loss = (w * nll).sum() / w.sum().clamp(min=1.0)
        if self.z_loss > 0.0:
            lse = torch.logsumexp(logits, dim=-1)
            loss = loss + self.z_loss * (lse.pow(2) * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
        return loss


@torch.no_grad()
def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, kinds: torch.Tensor) -> dict:
    """Per-kind greedy accuracy, useful for monitoring (structural / pointer / exponent / digits)."""
    pred = logits.argmax(-1)
    correct = (pred == targets).float()
    out = {}
    groups = {
        "struct": kinds == KIND_STRUCT,
        "tgt_ptr": kinds == KIND_TGT,
        "exponent": kinds == KIND_EXP,
        "digit1": kinds == KIND_DIGIT,
        "digits_rest": kinds > KIND_DIGIT,
    }
    for name, mask in groups.items():
        if mask.any():
            out[f"acc/{name}"] = correct[mask].mean().item()
    return out
