"""ToR-OT: token regression for optimal transport with an encoder-decoder transformer.

Package layout
--------------
data/        synthetic instance generation (Hugging Face dataset), loading, batching
tokenizer.py serialization of sparse plans into tokens (exponent-first scientific notation)
model.py     set encoder over bins + autoregressive decoder with pointer heads
loss.py      significance-weighted, ordinal cross-entropy
decode.py    grammar-constrained greedy generation with KV cache, transpose averaging
metrics.py   solution-invariant and plan-level evaluation metrics
train.py     training entry point (single GPU or torchrun DDP)
evaluate.py  evaluation entry point
"""

__version__ = "2.0.0"
