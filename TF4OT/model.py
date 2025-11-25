import torch
import torch.nn as nn
import numpy as np
import scipy as sp
import torch.nn.functional as F
from flash_attn import (
    flash_attn_qkvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)
from flash_attn.bert_padding import unpad_input, pad_input
from typing import Optional, Tuple

NEG_INF = float('-inf')

class FlashSelfAttention(nn.Module):
    """
    Drop-in, batch-first self-attention layer that mirrors nn.MultiheadAttention(..., batch_first=True)
    but internally uses FlashAttention (varlen) with qkv-packed kernels.

    Forward signature:
        forward(x, key_padding_mask=None, causal=True) -> [B, L, D]
    where key_padding_mask is torch.bool with True = PAD (torch convention).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.input_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,                          # [B, L, D]
        key_padding_mask: torch.Tensor | None = None,  # [B, L] bool (True = PAD)
        causal: bool = True
    ) -> torch.Tensor:
        assert x.ndim == 3, "expected [B, L, D]"
        B, L, D = x.shape
        assert D == self.embed_dim, "last dim must equal embed_dim"

        # FlashAttention requires CUDA tensors
        if not x.is_cuda:
            raise AssertionError("FlashAttention requires CUDA tensors. Move inputs/model to GPU.")

        # Pick a kernel-friendly dtype
        # Use bf16 on Ampere+ if you like; fp16 is usually safe everywhere FA is supported.
        attn_dtype = torch.bfloat16 if torch.cuda.get_device_capability(x.device)[0] >= 8 else torch.float16
        orig_dtype = x.dtype

        # qkv: [B, L, 3D]
        qkv = self.input_proj(x)
        if qkv.dtype not in (torch.float16, torch.bfloat16):
            qkv = qkv.to(attn_dtype)  # <- ensure FA input is fp16/bf16

        # Build keep mask: True = keep (valid), as expected by unpad_input
        if key_padding_mask is None:
            keep_mask = torch.ones((B, L), dtype=torch.bool, device=x.device)
        else:
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(torch.bool)
            keep_mask = ~key_padding_mask

        # Unpad to packed format
        # Returns (x_unpad, indices, cu_q_lens, max_seqlen, total_nnz)
        qkv_unpad, indices, cu_q_lens, max_s, _ = unpad_input(qkv, keep_mask)
        # Reshape to [T, 3, H, Hd]
        qkv_unpad = qkv_unpad.view(-1, 3, self.num_heads, self.head_dim)

        # FlashAttention kernel (dropout enabled only in train mode)
        p = self.dropout if self.training and self.dropout > 0.0 else 0.0
        y_unpad = flash_attn_varlen_qkvpacked_func(
            qkv_unpad,          # [T, 3, H, Hd]
            cu_q_lens,          # [B+1]
            max_s,              # int
            p,                  # dropout prob
            softmax_scale=None, # default 1/sqrt(Hd)
            causal=causal
        )                       # [T, H, Hd] collapsed to [T, D] after reshape

        y_unpad = y_unpad.reshape(-1, D)          # [T, D]
        y = pad_input(y_unpad, indices, B, L)     # [B, L, D]

        # 🔧 Cast input to out_proj’s weight dtype to avoid matmul dtype mismatch
        proj_in = y.to(self.out_proj.weight.dtype, non_blocking=True)  # typically float32
        out = self.out_proj(proj_in)  # matmul now has matching dtypes


        # out_proj is typically fp32; PyTorch will promote as needed.
        #out = self.out_proj(y)                                # usually fp32 result

        # Return to the caller’s original dtype so residuals match `seq`
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out

class TokenRegressor_flash(nn.Module):
    """
      Transformer-based model for token regression. FlashAttention version.
    """
    def __init__(self,
                 tokenizer,
                 input_cost_type,
                 encoder_ab: nn.Module,
                 encoder_c: nn.Module = None,
                 emb_dim_ab: int = 128,
                 emb_dim_c: int = 128,
                 emb_dim_d: int = 128,
                 max_seq_len: int = 1024,
                 num_transformer_layers: int = 1,
                 num_attention_heads:    int = 1,
                 dropout: float = 0.0):

        super().__init__()

        self.tokenizer  = tokenizer
        self.cost_type  = input_cost_type
        self.encoder_ab = encoder_ab
        self.encoder_c  = encoder_c
        self.emb_dim_ab = emb_dim_ab
        self.emb_dim_c  = emb_dim_c
        self.emb_dim_d  = emb_dim_d
        self.max_seq_len= max_seq_len
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads    = num_attention_heads
        self.dropout = dropout

        # token + positional embeddings
        self.token_emb = nn.Embedding(self.tokenizer.vocab_size, self.emb_dim_d)  # padding_idx=tokenizer.pad_id
        self.pos_emb   = nn.Embedding(self.max_seq_len, self.emb_dim_d)           # sequence dimension
        # projection layer to align concatenated prefix back to embedding dimension
        if self.cost_type == 'variable':
            # cost variable
            self.prefix_proj = nn.Linear(self.emb_dim_ab + self.emb_dim_c, self.emb_dim_d)
        elif self.cost_type == 'semivariable':
            # cost semivariable
            self.prefix_proj = nn.Linear(self.emb_dim_ab + self.emb_dim_c, self.emb_dim_d)
        elif self.cost_type == 'fixed':
            # cost fixed
            self.prefix_proj = nn.Linear(self.emb_dim_ab, self.emb_dim_d)

        # stacked transformer blocks
        self.transformer_layers = nn.ModuleList()
        for _ in range(self.num_transformer_layers):

            attn = FlashSelfAttention(
                embed_dim=self.emb_dim_d,
                num_heads=self.num_attention_heads,
                dropout=self.dropout
            )
            ffn = nn.Sequential(nn.Linear(self.emb_dim_d, 4*self.emb_dim_d),
                                nn.GELU(),
                                nn.Linear(4*self.emb_dim_d, self.emb_dim_d))

            layernorm1 = nn.LayerNorm(self.emb_dim_d, eps=1e-6)
            layernorm2 = nn.LayerNorm(self.emb_dim_d, eps=1e-6)

            self.transformer_layers.append(nn.ModuleList([attn, ffn, layernorm1, layernorm2]))

        # final projection → vocab logits
        self.out_proj = nn.Linear(self.emb_dim_d, self.tokenizer.vocab_size)

        # KV cache
        #self.window_size = self.max_seq_len
        #self.register_buffer("cache_k", None, persistent=False)
        #self.register_buffer("cache_v", None, persistent=False)

    def forward(self,
                encoder_input_ab,
                decoded_ids,
                encoder_input_c=None,
                pad_mask=None):
        """
        Returns probabilities over the next token.

        X=encoder_input (Batch, Feature): [B, F]
        Y=decoded_ids (Batch, Sequence Length) : [B, L-1]  (all but last token)
        units (int): hidden dimension
        returns logits over the vocabulary (Batch, Sequence Length, Vocabulary Size): [B, L, V]
        """

        # 1) prefix = encode "x" 
        prefix_ab = self.encoder_ab(encoder_input_ab)
        prefix_ab = prefix_ab.unsqueeze(1)

        if self.cost_type == 'semivariable':
            # cost semivariable: encoder for one-hot cost vector
            prefix_c = self.encoder_c(encoder_input_c)
            prefix_c = prefix_c.unsqueeze(1)
            prefix = torch.cat([prefix_ab, prefix_c], dim=2)
        elif self.cost_type == 'variable':
            # cost variable: encoder for the variable - n*n cost matrix
            prefix_c = self.encoder_c(encoder_input_c)
            prefix_c = prefix_c.unsqueeze(1)
            # concat prefix_ab and prefix_c along feature dim -> [B, 1, 2*units]
            prefix = torch.cat([prefix_ab, prefix_c], dim=2)
        elif self.cost_type == 'fixed':
            # cost fixed
            prefix = prefix_ab  # [B, 1, units]
        # Project back to embedding dimension -> [B, 1, emb_dim_d]
        prefix = self.prefix_proj(prefix)

        # 2) embed decoder inputs
        decoded = self.token_emb(decoded_ids)      # [B, L-1, units]
        seq = torch.cat([prefix, decoded], dim=1)  # [B, L, units], L = L-1+1

        # 3) add positional embeddings
        batch_size, L, _ = seq.shape
        positions = torch.arange(start=0, end=L, step=1, device=seq.device)
        pos_embeddings = self.pos_emb(positions).unsqueeze(0)   # [1, L, units]
        seq = seq + pos_embeddings                              # broadcast→[B,L,units]

        # 4) Align pad_mask to [B, L] (True = PAD)
        if pad_mask is not None:
            if pad_mask.dtype != torch.bool:
                pad_mask = pad_mask.to(torch.bool)
            if pad_mask.shape[1] == L - 1:
                prefix_pad = torch.zeros((batch_size, 1), dtype=torch.bool, device=seq.device)
                pad_mask = torch.cat([prefix_pad, pad_mask], dim=1)
            assert pad_mask.shape == (batch_size, L)

        # 4) Transformer stack with causal masking
        for attn, ffn, layernorm1, layernorm2 in self.transformer_layers:
            attn_out = attn(seq, key_padding_mask=pad_mask, causal=True)   # [batch_size, L, D]
            seq = layernorm1(seq + attn_out)
            seq = layernorm2(seq + ffn(seq))

        # 5) Project to logits
        logits = self.out_proj(seq)
        return logits

    @torch.no_grad()
    def generate(self,
            encoder_input_ab,
            encoder_input_c=None,
            temperature: float = 1.0,
            top_k: int | None = None,
            top_p: float | None = None
        ):

        batch_size = encoder_input_ab.shape[0]
        # pre-allocate output token ids
        #token_ids = -1 * np.ones((batch_size, self.max_seq_len), dtype=int)
        token_ids = torch.full(
            size=(batch_size, self.max_seq_len),
            fill_value=-1,
            dtype=torch.long,
            device=encoder_input_ab.device
        )
        print("Starting generation...")
        for i in range(self.max_seq_len):
            #print (f"Generating token {i+1}/{self.max_seq_len}", end='\r')
            # tokens generated so far
            tensor_slice = token_ids[:, :i]
            #tensor_slice = torch.from_numpy(token_ids[:, :i]).to(device=encoder_input_ab[0].device, dtype=torch.long)
            logits = self.forward(encoder_input_ab,
                                  tensor_slice,
                                  encoder_input_c=encoder_input_c,
                                  pad_mask=None)       # [B, i+1, V], torch.Tensor (B: batch size, i+1: sequence length, V: vocabulary size)
            current_logits = logits[:, -1, :]          # [B, V], torch.Tensor

            #logits = logits.detach().cpu().numpy()          # shape [B, i+1, V]
            #current_logits = current_logits.detach().cpu().numpy()  # shape [B, V]

            # Apply top-k and top-p filtering
            if top_k is not None:
                top_k = min(top_k, current_logits.shape[-1])  # prevent index errors
                thresholds = np.sort(current_logits, axis=-1)[:, -top_k][:, np.newaxis]
                current_logits = np.where(current_logits < thresholds, NEG_INF, current_logits)

            if top_p is not None:

                probs = sp.special.softmax(current_logits / temperature, axis=-1)
                sorted_probs = np.sort(probs, axis=-1)[:, ::-1]
                cumulative_probs = np.cumsum(sorted_probs, axis=-1)
                cutoff_indices = np.argmax(cumulative_probs > top_p, axis=-1)

                for batch_idx in range(batch_size):
                    sorted_indices = np.argsort(current_logits[batch_idx, :])
                    top_k_indices = sorted_indices[: -cutoff_indices[batch_idx]]
                    current_logits[batch_idx, top_k_indices] = NEG_INF

            # [B, V]
            #probs = sp.special.softmax(current_logits / temperature, axis=-1)
            probs = F.softmax(current_logits / temperature, dim=-1)

            # Sample tokens.
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]
            token_ids[:, i] = next_token

            # sampled_ids = vectorized_sample(probs)
            # token_ids[:, i] = np.array(sampled_ids)

        return token_ids


class TokenRegressor(nn.Module):
    """
      Transformer-based model for token regression.
    """
    def __init__(self,
                 tokenizer,
                 input_cost_type,
                 encoder_ab: nn.Module,
                 encoder_c: nn.Module = None,
                 emb_dim_ab: int = 128,
                 emb_dim_c: int = 128,
                 emb_dim_d: int = 128,
                 max_seq_len: int = 1024,
                 num_transformer_layers: int = 1,
                 num_attention_heads:    int = 1,
                 dropout: float = 0.0):

        super().__init__()

        self.tokenizer  = tokenizer
        self.cost_type  = input_cost_type
        self.encoder_ab = encoder_ab
        self.encoder_c  = encoder_c
        self.emb_dim_ab = emb_dim_ab
        self.emb_dim_c  = emb_dim_c
        self.emb_dim_d  = emb_dim_d
        self.max_seq_len= max_seq_len
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads    = num_attention_heads
        self.dropout = dropout

        # token + positional embeddings
        self.token_emb = nn.Embedding(self.tokenizer.vocab_size, self.emb_dim_d)  # padding_idx=tokenizer.pad_id
        self.pos_emb   = nn.Embedding(self.max_seq_len, self.emb_dim_d)           # sequence dimension
        # projection layer to align concatenated prefix back to embedding dimension
        if encoder_c is not None:
            self.prefix_proj = nn.Linear(self.emb_dim_ab + self.emb_dim_c, self.emb_dim_d)
        else:
            self.prefix_proj = nn.Linear(self.emb_dim_ab, self.emb_dim_d)

        # stacked transformer blocks
        self.transformer_layers = nn.ModuleList()
        for _ in range(self.num_transformer_layers):

            # batch_first=True makes the layer expect (B, L, D) shaped inputs (batch, seq, feature).
            attn = nn.MultiheadAttention(num_heads=self.num_attention_heads,
                                         embed_dim=self.emb_dim_d,
                                         dropout=self.dropout,
                                         batch_first=True)

            ffn = nn.Sequential(nn.Linear(self.emb_dim_d, 4*self.emb_dim_d),
                                nn.GELU(),
                                nn.Linear(4*self.emb_dim_d, self.emb_dim_d))

            layernorm1 = nn.LayerNorm(self.emb_dim_d, eps=1e-6)
            layernorm2 = nn.LayerNorm(self.emb_dim_d, eps=1e-6)

            self.transformer_layers.append(nn.ModuleList([attn, ffn, layernorm1, layernorm2]))

        # final projection → vocab logits
        self.out_proj = nn.Linear(self.emb_dim_d, self.tokenizer.vocab_size)

        # KV cache
        #self.window_size = self.max_seq_len
        #self.register_buffer("cache_k", None, persistent=False)
        #self.register_buffer("cache_v", None, persistent=False)

    def forward(self,
                encoder_input_ab,
                decoded_ids,
                encoder_input_c=None,
                pad_mask=None):
        """
        Returns probabilities over the next token.

        X=encoder_input (Batch, Feature): [B, F]
        Y=decoded_ids (Batch, Sequence Length) : [B, L-1]  (all but last token)
        units (int): hidden dimension
        returns logits over the vocabulary (Batch, Sequence Length, Vocabulary Size): [B, L, V]
        """

        # 1) prefix = encode "x" 
        prefix_ab = self.encoder_ab(encoder_input_ab)
        prefix_ab = prefix_ab.unsqueeze(1)

        if encoder_input_c is not None:
            prefix_c = self.encoder_c(encoder_input_c)
            prefix_c = prefix_c.unsqueeze(1)
            # concat prefix_ab and prefix_C along feature dim -> [B, 1, 2*units]
            prefix = torch.cat([prefix_ab, prefix_c], dim=2)
        else:
            prefix = prefix_ab  # [B, 1, units]
        # Project back to embedding dimension -> [B, 1, emb_dim_d]
        prefix = self.prefix_proj(prefix)

        # 2) embed decoder inputs
        decoded = self.token_emb(decoded_ids)      # [B, L-1, units]
        seq = torch.cat([prefix, decoded], dim=1)  # [B, L, units], L = L-1+1

        # 3) add positional embeddings
        L = seq.size(1)
        positions = torch.arange(start=0, end=L, step=1, device=seq.device)
        pos_embeddings = self.pos_emb(positions).unsqueeze(0)   # [1, L, units]
        seq = seq + pos_embeddings                              # broadcast→[B,L,units]

        # 4) Create causal mask for self-attention.
        #    transformer stack with causal mask
        #    mask shape: [L, L], True = masked
        causal_mask = torch.triu(torch.ones(L, L, device=seq.device), diagonal=1).bool()

        # Apply transformer layers.
        for attn, ffn, layernorm1, layernorm2 in self.transformer_layers:
            # PyTorch MHA returns (output, weights). We only need the output.
            # It expects query, key, value, and an optional attention mask.
            attn_out, _ = attn(seq, seq, seq, attn_mask=causal_mask, need_weights=False, key_padding_mask=pad_mask)
            seq = layernorm1(seq + attn_out) # Add & Norm
            ffn_out = ffn(seq)
            seq = layernorm2(seq + ffn_out) # Add & Norm

        # 5) project to vocab logits
        logits = self.out_proj(seq)  # [B, L, V]
        return logits

    @torch.no_grad()
    def generate(self,
            encoder_input_ab,
            encoder_input_c=None,
            temperature: float = 1.0,
            top_k: int | None = None,
            top_p: float | None = None
        ):

        batch_size = encoder_input_ab.shape[0]
        # pre-allocate output token ids
        #token_ids = -1 * np.ones((batch_size, self.max_seq_len), dtype=int)
        token_ids = torch.full(
            size=(batch_size, self.max_seq_len),
            fill_value=-1,
            dtype=torch.long,
            device=encoder_input_ab.device
        )
        print("Starting generation...")
        for i in range(self.max_seq_len):
            print (f"Generating token {i+1}/{self.max_seq_len}", end='\r')
            # tokens generated so far
            tensor_slice = token_ids[:, :i]
            #tensor_slice = torch.from_numpy(token_ids[:, :i]).to(device=encoder_input_ab[0].device, dtype=torch.long)
            logits = self.forward(encoder_input_ab,
                                  tensor_slice,
                                  encoder_input_c=encoder_input_c,
                                  pad_mask=None)       # [B, i+1, V], torch.Tensor (B: batch size, i+1: sequence length, V: vocabulary size)
            current_logits = logits[:, -1, :]          # [B, V], torch.Tensor

            #logits = logits.detach().cpu().numpy()          # shape [B, i+1, V]
            #current_logits = current_logits.detach().cpu().numpy()  # shape [B, V]

            # Apply top-k and top-p filtering
            if top_k is not None:
                top_k = min(top_k, current_logits.shape[-1])  # prevent index errors
                thresholds = np.sort(current_logits, axis=-1)[:, -top_k][:, np.newaxis]
                current_logits = np.where(current_logits < thresholds, NEG_INF, current_logits)

            if top_p is not None:

                probs = sp.special.softmax(current_logits / temperature, axis=-1)
                sorted_probs = np.sort(probs, axis=-1)[:, ::-1]
                cumulative_probs = np.cumsum(sorted_probs, axis=-1)
                cutoff_indices = np.argmax(cumulative_probs > top_p, axis=-1)

                for batch_idx in range(batch_size):
                    sorted_indices = np.argsort(current_logits[batch_idx, :])
                    top_k_indices = sorted_indices[: -cutoff_indices[batch_idx]]
                    current_logits[batch_idx, top_k_indices] = NEG_INF

            # [B, V]
            #probs = sp.special.softmax(current_logits / temperature, axis=-1)
            probs = F.softmax(current_logits / temperature, dim=-1)

            # Sample tokens.
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]
            token_ids[:, i] = next_token

            # sampled_ids = vectorized_sample(probs)
            # token_ids[:, i] = np.array(sampled_ids)

        return token_ids


class TokenRegressor_two_prefix_tokens(nn.Module):
    """
      Transformer-based model for token regression with two prefix tokens.
    """
    def __init__(self,
                 tokenizer,
                 input_cost_type,
                 encoder_ab:    nn.Module,
                 encoder_c:     nn.Module = None,
                 emb_dim_ab:    int = 128,
                 emb_dim_c:     int = 128,
                 emb_dim_d:     int = 128,
                 max_seq_len:   int = 1024,
                 num_transformer_layers: int = 1,
                 num_attention_heads:    int = 1,
                 dropout: float = 0.0,
                 ):

        super().__init__()

        self.tokenizer  = tokenizer
        self.cost_type  = input_cost_type
        self.encoder_ab = encoder_ab
        self.encoder_c  = encoder_c
        self.emb_dim_ab = emb_dim_ab
        self.emb_dim_c  = emb_dim_c
        self.emb_dim_d  = emb_dim_d
        self.max_seq_len= max_seq_len
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads    = num_attention_heads
        self.dropout = dropout

        # token + positional embeddings))
        self.token_emb = nn.Embedding(self.tokenizer.vocab_size, self.emb_dim_d) # padding_idx=tokenizer.pad_id)
        self.pos_emb   = nn.Embedding(self.max_seq_len + 2, self.emb_dim_d)      # sequence dimension; allow for 2 additional tokens (prefixes)
        # projection layer to align concatenated prefix back to embedding dimension
        # if encoder_c is not None:
        #     self.prefix_proj = nn.Linear(self.emb_dim_ab + self.emb_dim_c, self.emb_dim_d)
        # else:
        #     self.prefix_proj = nn.Linear(self.emb_dim_ab, self.emb_dim_d)

        # stacked transformer blocks
        self.transformer_layers = nn.ModuleList()
        for _ in range(self.num_transformer_layers):

            # batch_first=True makes the layer expect (B, L, D) shaped inputs (batch, seq, feature).
            attn = nn.MultiheadAttention(num_heads=self.num_attention_heads,
                                         embed_dim=self.emb_dim_d,
                                         dropout=self.dropout,
                                         batch_first=True)

            ffn = nn.Sequential(nn.Linear(self.emb_dim_d, 4*self.emb_dim_d),
                                nn.GELU(),
                                nn.Linear(4*self.emb_dim_d, self.emb_dim_d))

            layernorm1 = nn.LayerNorm(self.emb_dim_d, eps=1e-6)
            layernorm2 = nn.LayerNorm(self.emb_dim_d, eps=1e-6)

            self.transformer_layers.append(nn.ModuleList([attn, ffn, layernorm1, layernorm2]))

        # final projection → vocab logits
        self.out_proj = nn.Linear(self.emb_dim_d, self.tokenizer.vocab_size)

        # KV cache
        #self.window_size = self.max_seq_len
        #self.register_buffer("cache_k", None, persistent=False)
        #self.register_buffer("cache_v", None, persistent=False)

    def forward(self,
                encoder_input_ab,
                decoded_ids,
                encoder_input_c=None,
                pad_mask=None):
        """
        Returns probabilities over the next token.

        X=encoder_input (Batch, Feature): [B, F]
        Y=decoded_ids (Batch, Sequence Length) : [B, L-1]  (all but last token)
        units (int): hidden dimension
        returns logits over the vocabulary (Batch, Sequence Length, Vocabulary Size): [B, L, V]
        """
        decoded = self.token_emb(decoded_ids)      # [B, L-1, units]

        # 1) prefix = encode "x" 
        prefix_ab = self.encoder_ab(encoder_input_ab)
        prefix_ab = prefix_ab.unsqueeze(1)

        if encoder_input_c is not None:
            prefix_c = self.encoder_c(encoder_input_c)
            prefix_c = prefix_c.unsqueeze(1)
            # sequence: two prefixes + decoded tokens
            # 2) concat prefix_ab and prefix_c along feature dim -> [B, 1, emb_dim_d]
            seq = torch.cat([prefix_ab, prefix_c, decoded], dim=1)  # [B, L+1, emb_dim_d], L+1 = 2 + L-1

        else:
            # 2) concat prefix_ab and decoded along feature dim -> [B, L, units], L = L-1+1
            seq = torch.cat([prefix_ab, decoded], dim=1)  # [B, L, units], L = L-1+1


        # 3) add positional embeddings
        L = seq.size(1)
        positions      = torch.arange(start=0, end=L, step=1, device=seq.device)
        pos_embeddings = self.pos_emb(positions).unsqueeze(0)   # [1, L, units]
        seq = seq + pos_embeddings                              # broadcast→[B,L,units]

        # 4) Create causal mask for self-attention.
        #    transformer stack with causal mask
        #    mask shape: [L, L], True = masked
        causal_mask = torch.triu(torch.ones(L, L, device=seq.device), diagonal=1).bool()

        # Apply transformer layers.
        for attn, ffn, layernorm1, layernorm2 in self.transformer_layers:
            # PyTorch MHA returns (output, weights). We only need the output.
            # It expects query, key, value, and an optional attention mask.
            attn_out, _ = attn(seq, seq, seq, attn_mask=causal_mask, need_weights=False, key_padding_mask=pad_mask)
            seq = layernorm1(seq + attn_out) # Add & Norm
            ffn_out = ffn(seq)
            seq = layernorm2(seq + ffn_out) # Add & Norm

        # 5) project to vocab logits
        logits = self.out_proj(seq)  # [B, L, V]
        return logits

    @torch.no_grad()
    def generate(self,
            encoder_input_ab,
            encoder_input_c=None,
            temperature: float = 1.0,
            top_k: int | None = None,
            top_p: float | None = None
        ):

        batch_size = encoder_input_ab.shape[0]
        # pre-allocate output token ids
        #token_ids = -1 * np.ones((batch_size, self.max_seq_len), dtype=int)
        token_ids = torch.full(
            size=(batch_size, self.max_seq_len),
            fill_value=-1,
            dtype=torch.long,
            device=encoder_input_ab.device
        )
        print("Starting generation...")
        for i in range(self.max_seq_len):
            print (f"Generating token {i+1}/{self.max_seq_len}", end='\r')
            # tokens generated so far
            tensor_slice = token_ids[:, :i]
            #tensor_slice = torch.from_numpy(token_ids[:, :i]).to(device=encoder_input_ab[0].device, dtype=torch.long)
            logits = self.forward(encoder_input_ab,
                                  tensor_slice,
                                  encoder_input_c=encoder_input_c,
                                  pad_mask=None)       # [B, i+1, V], torch.Tensor (B: batch size, i+1: sequence length, V: vocabulary size)
            current_logits = logits[:, -1, :]          # [B, V], torch.Tensor

            #logits = logits.detach().cpu().numpy()          # shape [B, i+1, V]
            #current_logits = current_logits.detach().cpu().numpy()  # shape [B, V]

            # Apply top-k and top-p filtering
            if top_k is not None:
                top_k = min(top_k, current_logits.shape[-1])  # prevent index errors
                thresholds = np.sort(current_logits, axis=-1)[:, -top_k][:, np.newaxis]
                current_logits = np.where(current_logits < thresholds, NEG_INF, current_logits)

            if top_p is not None:

                probs = sp.special.softmax(current_logits / temperature, axis=-1)
                sorted_probs = np.sort(probs, axis=-1)[:, ::-1]
                cumulative_probs = np.cumsum(sorted_probs, axis=-1)
                cutoff_indices = np.argmax(cumulative_probs > top_p, axis=-1)

                for batch_idx in range(batch_size):
                    sorted_indices = np.argsort(current_logits[batch_idx, :])
                    top_k_indices = sorted_indices[: -cutoff_indices[batch_idx]]
                    current_logits[batch_idx, top_k_indices] = NEG_INF

            # [B, V]
            #probs = sp.special.softmax(current_logits / temperature, axis=-1)
            probs = F.softmax(current_logits / temperature, dim=-1)

            # Sample tokens.
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]
            token_ids[:, i] = next_token

            # sampled_ids = vectorized_sample(probs)
            # token_ids[:, i] = np.array(sampled_ids)

        return token_ids