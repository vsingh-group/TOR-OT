"""Encoder-decoder transformer for sparse OT plan generation.

Encoder (permutation-equivariant set encoder, symmetry 6.2):
    tokens = [cost-type] + [source bins] + [target bins]
    bin token = type embedding + mass features + Fourier features of the bin coordinate.
    There is no sequence positional encoding: position enters only through coordinates,
    so relabelling the bins (with their coordinates attached) permutes the encoder
    states and, through the pointer heads, the predicted plan.

Row decoding (``cfg.decoding == "row"``): the encoder sees the *remaining* problem, i.e. the
    source bins not yet decoded and the remaining column capacities as target masses. The
    first source bin is the row being generated and carries a learned marker.

Decoder (causal, RoPE, cross-attention to the encoder):
    Output logits are the concatenation of vocabulary logits and *pointer* logits
    (dot products with the target-bin encoder states), so column-index prediction is
    size-agnostic. Pointer tokens are embedded from the encoder state they point to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torot.config import ModelConfig
from torot.data.cost import COST_TYPES

LayerCache = Dict[str, torch.Tensor]


# --------------------------------------------------------------------------- building blocks
class RMSNorm(nn.Module):
    """Fused RMS normalisation over the last dimension (``F.rms_norm`` keeps far fewer
    intermediates for backward than a hand-written float32 implementation)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.dropout(F.silu(self.w_gate(x)) * self.w_up(x)))


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """x: [B, H, T, hd]; rotates positions offset .. offset + T - 1."""
        T = x.shape[2]
        cos = self.cos[offset : offset + T].to(x.dtype)
        sin = self.sin[offset : offset + T].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """Multi-head attention with optional cross-attention context, RoPE and KV cache.

    Queries and keys are RMS-normalised per head (QK-norm) so attention logits stay
    bounded; this removes the logit-growth instability seen without it.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        rope: Optional[RotaryEmbedding] = None,
        cache: Optional[LayerCache] = None,
        offset: int = 0,
    ) -> torch.Tensor:
        """x: [B, T, D] queries. ctx: [B, S, D] keys/values (None -> self-attention).
        key_mask: [B, S] bool, True = attend. cache: per-layer dict, updated in place."""
        q = self.q_norm(self._split(self.wq(x)))
        cross = ctx is not None
        if cross and cache is not None and "k" in cache:
            k, v = cache["k"], cache["v"]
        else:
            src = ctx if cross else x
            k, v = self.k_norm(self._split(self.wk(src))), self._split(self.wv(src))
            if rope is not None and not cross:
                k = rope(k, offset)
            if cache is not None:
                if not cross and "k" in cache:
                    k = torch.cat([cache["k"], k], dim=2)
                    v = torch.cat([cache["v"], v], dim=2)
                cache["k"], cache["v"] = k, v
        if rope is not None and not cross:
            q = rope(q, offset)

        attn_mask = key_mask[:, None, None, :] if key_mask is not None else None
        use_causal = causal and not cross and q.shape[2] > 1
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=use_causal
        )
        B, _, T, _ = out.shape
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_mask=key_mask)
        return x + self.ffn(self.norm2(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.self_attn = Attention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = RMSNorm(cfg.d_model)
        self.cross_attn = Attention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm3 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor,
        enc_mask: torch.Tensor,
        rope: RotaryEmbedding,
        cache: Optional[Dict[str, LayerCache]] = None,
        offset: int = 0,
    ) -> torch.Tensor:
        self_cache = cache["self"] if cache is not None else None
        cross_cache = cache["cross"] if cache is not None else None
        x = x + self.self_attn(self.norm1(x), causal=True, rope=rope, cache=self_cache, offset=offset)
        x = x + self.cross_attn(self.norm2(x), ctx=enc, key_mask=enc_mask, cache=cross_cache)
        return x + self.ffn(self.norm3(x))


# --------------------------------------------------------------------------- input features
class FourierFeatures(nn.Module):
    """[sin(2^k pi x), cos(2^k pi x)] for k < bands, per coordinate, plus the raw coordinate."""

    def __init__(self, coord_dim: int, bands: int):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(bands)) * math.pi, persistent=False)
        self.out_dim = coord_dim * (2 * bands + 1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        ang = coords.unsqueeze(-1) * self.freqs                    # [..., coord_dim, bands]
        feats = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)
        return torch.cat([coords, feats], dim=-1)


def mass_features(mass: torch.Tensor) -> torch.Tensor:
    """Bounded features spanning the dynamic range of masses: [m, sqrt(m), log-scaled m]."""
    log_term = 1.0 + torch.log10(mass.clamp(min=1e-9)) / 9.0        # 0 for 1e-9 .. 1 for 1
    return torch.stack([mass, mass.sqrt(), log_term], dim=-1)


@dataclass
class EncoderOutput:
    states: torch.Tensor      # [B, 1 + n_pad + m_pad, D]  (cost token, source bins, target bins)
    key_mask: torch.Tensor    # [B, 1 + n_pad + m_pad]
    tgt_states: torch.Tensor  # [B, m_pad, D]  target-bin states, the pointer targets
    tgt_mask: torch.Tensor    # [B, m_pad]


# --------------------------------------------------------------------------- model
class ToROT(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int, coord_dim: int = 2):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        D = cfg.d_model

        # encoder inputs
        self.fourier = FourierFeatures(coord_dim, cfg.fourier_bands)
        self.bin_type_emb = nn.Embedding(2, D)                       # 0 = source, 1 = target
        self.mass_proj = nn.Linear(3, D)
        self.coord_proj = nn.Linear(self.fourier.out_dim, D)
        self.cost_emb = nn.Embedding(len(COST_TYPES), D)
        if cfg.decoding not in ("plan", "row"):
            raise ValueError(f"model.decoding must be 'plan' or 'row', got {cfg.decoding!r}")
        if cfg.decoding == "row":
            self.current_emb = nn.Parameter(torch.randn(D) * 0.02)   # marks the row being generated
        self.enc_layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.enc_layers))
        self.enc_norm = RMSNorm(D)

        # decoder inputs
        self.tok_emb = nn.Embedding(vocab_size, D)
        self.ptr_in_proj = nn.Linear(D, D)
        self.rope = RotaryEmbedding(D // cfg.n_heads, cfg.max_seq_len)
        self.dec_layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.dec_layers))
        self.dec_norm = RMSNorm(D)

        # output heads (pointer query/key are RMS-normalised: bounded pointer logits)
        self.lm_head = nn.Linear(D, vocab_size, bias=False)
        self.ptr_q = nn.Linear(D, D, bias=False)
        self.ptr_k = nn.Linear(D, D, bias=False)
        self.ptr_q_norm = RMSNorm(D)
        self.ptr_k_norm = RMSNorm(D)
        self.dropout = nn.Dropout(cfg.dropout)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    # ------------------------------------------------------------------ encoder
    def encode(self, batch: Dict[str, torch.Tensor]) -> EncoderOutput:
        a, b = batch["a"], batch["b"]
        B, n_pad = a.shape
        src = (
            self.bin_type_emb.weight[0]
            + self.mass_proj(mass_features(a))
            + self.coord_proj(self.fourier(batch["coords_src"]))
        )
        tgt = (
            self.bin_type_emb.weight[1]
            + self.mass_proj(mass_features(b))
            + self.coord_proj(self.fourier(batch["coords_tgt"]))
        )
        if self.cfg.decoding == "row":
            first = torch.zeros(n_pad, 1, device=a.device, dtype=src.dtype)
            first[0] = 1.0
            src = src + first * self.current_emb
        cost = self.cost_emb(batch["cost_type"]).unsqueeze(1)
        x = self.dropout(torch.cat([cost, src, tgt], dim=1))
        key_mask = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=a.device), batch["src_mask"], batch["tgt_mask"]], dim=1
        )
        for layer in self.enc_layers:
            x = layer(x, key_mask)
        x = self.enc_norm(x)
        return EncoderOutput(
            states=x,
            key_mask=key_mask,
            tgt_states=x[:, 1 + n_pad :],
            tgt_mask=batch["tgt_mask"],
        )

    # ------------------------------------------------------------------ decoder
    def embed_tokens(self, ids: torch.Tensor, enc: EncoderOutput) -> torch.Tensor:
        """Vocabulary ids use the token table; pointer ids use the pointed-to target-bin state."""
        is_ptr = ids >= self.vocab_size
        vocab_ids = torch.where(is_ptr, torch.zeros_like(ids), ids)
        emb = self.tok_emb(vocab_ids)

        tgt_idx = (ids - self.vocab_size).clamp(min=0)
        gathered = torch.gather(enc.tgt_states, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, enc.tgt_states.shape[-1]))
        return torch.where(is_ptr.unsqueeze(-1), self.ptr_in_proj(gathered), emb)

    def output_logits(self, h: torch.Tensor, enc: EncoderOutput) -> torch.Tensor:
        """[B, T, V + m_pad]: vocabulary logits followed by target-bin pointer logits."""
        vocab_logits = self.lm_head(h)
        q = self.ptr_q_norm(self.ptr_q(h))
        k = self.ptr_k_norm(self.ptr_k(enc.tgt_states))
        ptr_logits = torch.einsum("btd,bsd->bts", q, k) / math.sqrt(h.shape[-1])
        ptr_logits = ptr_logits.masked_fill(~enc.tgt_mask[:, None, :], float("-inf"))
        return torch.cat([vocab_logits, ptr_logits], dim=-1)

    def decode(
        self,
        ids: torch.Tensor,
        enc: EncoderOutput,
        cache: Optional[List[Dict[str, LayerCache]]] = None,
        offset: int = 0,
    ) -> torch.Tensor:
        x = self.dropout(self.embed_tokens(ids, enc))
        for i, layer in enumerate(self.dec_layers):
            x = layer(x, enc.states, enc.key_mask, self.rope, cache=None if cache is None else cache[i], offset=offset)
        return self.output_logits(self.dec_norm(x), enc)

    def new_cache(self) -> List[Dict[str, LayerCache]]:
        return [{"self": {}, "cross": {}} for _ in self.dec_layers]

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Teacher-forced logits [B, T, E] for ``batch['input_ids']``."""
        enc = self.encode(batch)
        return self.decode(batch["input_ids"], enc)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
