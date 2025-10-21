import torch
import torch.nn as nn
import numpy as np
import scipy as sp

NEG_INF = float('-inf')

def sample_from_probs(prob_vector):
  return np.random.choice(range(len(prob_vector)), p=prob_vector)
vectorized_sample = np.vectorize(sample_from_probs, signature='(n)->()')

class TokenRegressor(nn.Module):
    """
      Transformer-based model for token regression.
    """
    def __init__(self,
                 tokenizer,
                 encoder_ab: nn.Module,
                 encoder_C: nn.Module = None,
                 emb_dim: int = 128,
                 seq_dim: int = 1024,
                 num_transformer_layers: int = 1,
                 num_attention_heads:    int = 1,
                 dropout: float = 0.0):
    
        super().__init__()

        self.tokenizer  = tokenizer
        self.encoder_ab = encoder_ab
        self.encoder_C  = encoder_C
        self.emb_dim    = emb_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads    = num_attention_heads
        self.dropout = dropout

        # token + positional embeddings
        self.token_emb = nn.Embedding(self.tokenizer.vocab_size, self.emb_dim, padding_idx=tokenizer.pad_id)
        self.pos_emb   = nn.Embedding(seq_dim, self.emb_dim)
        # projection layer to align concatenated prefix back to embedding dimension
        if encoder_C is not None:
            self.prefix_proj = nn.Linear(2 * self.emb_dim, self.emb_dim)
        else:
            self.prefix_proj = None

        # stacked transformer blocks
        self.transformer_layers = nn.ModuleList()
        for _ in range(self.num_transformer_layers):

            # batch_first=True makes the layer expect (B, L, D) shaped inputs (batch, seq, feature).
            attn = nn.MultiheadAttention(num_heads=self.num_attention_heads,
                                         embed_dim=self.emb_dim,
                                         dropout=self.dropout,
                                         batch_first=True)

            ffn = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim),
                                nn.ReLU(),
                                nn.Linear(self.emb_dim, self.emb_dim))

            layernorm1 = nn.LayerNorm(self.emb_dim, eps=1e-6)
            layernorm2 = nn.LayerNorm(self.emb_dim, eps=1e-6)

            self.transformer_layers.append(nn.ModuleList([attn, ffn, layernorm1, layernorm2]))

        # final projection → vocab logits
        self.out_proj = nn.Linear(self.emb_dim, self.tokenizer.vocab_size)

    def forward(self,
                encoder_input_ab,
                decoded_ids,
                encoder_input_C=None,
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

        if encoder_input_C is not None:
            prefix_C = self.encoder_C(encoder_input_C)
            prefix_C = prefix_C.unsqueeze(1)
            # concat prefix_ab and prefix_C along feature dim -> [B, 1, 2*units]
            prefix = torch.cat([prefix_ab, prefix_C], dim=2)
            # Project back to embedding dimension -> [B, 1, emb_dim]
            prefix = self.prefix_proj(prefix)
        else:
            prefix = prefix_ab  # [B, 1, units]
        # 2) embed decoder inputs
        decoded = self.token_emb(decoded_ids)      # [B, L-1, units]
        seq = torch.cat([prefix, decoded], dim=1)  # [B, L, units], L = L-1+1

        # 3) add positional embeddings
        L = seq.size(1)
        #print("L: ", L)
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
            encoder_input_C=None,
            temperature: float = 1.0,
            top_k: int | None = None,
            top_p: float | None = None,
            max_seq_len: int = 2500):

        B = encoder_input_ab.shape[0]
        token_ids = -1 * np.ones((B, max_seq_len), dtype=int)

        print("Starting generation...")
        for i in range(max_seq_len):
            print (f"Generating token {i+1}/{max_seq_len}", end='\r')
            tensor_slice = torch.from_numpy(token_ids[:, :i]).to(device=encoder_input_ab[0].device, dtype=torch.long)
            logits = self.forward(encoder_input_ab,
                                  tensor_slice,
                                  encoder_input_C=None,
                                  pad_mask=None)       # [B, i+1, V], torch.Tensor
            current_logits = logits[:, -1, :]          # [B, V], torch.Tensor

            logits = logits.detach().cpu().numpy()          # shape [B, i+1, V]
            current_logits = current_logits.detach().cpu().numpy()  # shape [B, V]

            # Logit restriction: mask invalid tokens (create mask as torch.bool on the correct device)
            # current_logits[:, ~self._vocab.logit_mask(i)] = NEG_INF

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

                for batch_idx in range(B):
                    sorted_indices = np.argsort(current_logits[batch_idx, :])
                    top_k_indices = sorted_indices[: -cutoff_indices[batch_idx]]
                    current_logits[batch_idx, top_k_indices] = NEG_INF

            # [B, V]
            probs = sp.special.softmax(current_logits / temperature, axis=-1)

            # Sample tokens.
            sampled_ids = vectorized_sample(probs)
            token_ids[:, i] = np.array(sampled_ids)

        return token_ids