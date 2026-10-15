"""Structured configuration (OmegaConf) for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import OmegaConf


@dataclass
class DataConfig:
    repo_id: str = "ivanjaenm/torot_data_1M"      # uniform-on-simplex marginals, six distinct 1D costs; or a local save_to_disk dir
    cache_dir: Optional[str] = None
    train_split: str = "train"
    val_split: str = "val"
    max_bins: int = 200                 # upper bound on n and m (sanity check only)
    transpose_aug_prob: float = 0.5     # symmetry 6.1: swap (a, b) and transpose the plan
    residual_noise: float = 0.0         # row decoding: relative noise on the remaining capacities during training
    residual_leftover_prob: float = 0.0  # row decoding: chance that a used-up column shows a small fake leftover
    residual_leftover_range: List[float] = field(default_factory=lambda: [1e-6, 1e-2])  # log-uniform leftover size
    num_workers: int = 4


@dataclass
class TokenizerConfig:
    mantissa_digits: int = 4            # d1..d4 after the exponent token
    max_exponent: int = 9               # masses below 10^-max_exponent are dropped


@dataclass
class ModelConfig:
    d_model: int = 512
    n_heads: int = 8
    enc_layers: int = 6
    dec_layers: int = 12
    d_ff: int = 1408                    # ~8/3 * d_model, multiple of 64
    fourier_bands: int = 8
    dropout: float = 0.0
    max_seq_len: int = 4096
    decoding: str = "plan"              # "plan": one sequence per plan; "row": one sequence per row,
                                        # conditioned on the remaining column capacities


@dataclass
class LossConfig:
    digit_weight_decay: float = 0.5     # weight of mantissa digit p is decay ** (p - 1)
    exponent_weight: float = 1.0
    ordinal_eps: float = 0.05           # probability mass moved to each numeric neighbour
    z_loss: float = 1e-4                # coefficient of logsumexp(logits)^2 (logit-magnitude regulariser)


@dataclass
class OptimConfig:
    lr: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 50_000             # ~17 epochs of 1M instances on 4 GPUs at 65K tokens/GPU/step
    min_lr_ratio: float = 0.1
    batch_size: int = 512               # max instances per device batch (token budget binds first)
    max_tokens: int = 65536             # max padded decoder tokens per device batch (~33 GB on A100-40GB)
    grad_clip: float = 1.0
    ema_decay: float = 0.999


@dataclass
class RunConfig:
    out_dir: str = "experiments/torot/default"
    seed: int = 0
    log_every: int = 50
    val_every: int = 1000
    val_batches: int = 20               # batches used for validation loss
    val_decode_instances: int = 256     # instances decoded for validation metrics
    max_decode_len: int = 2048
    bf16: bool = True
    compile: bool = False
    resume: Optional[str] = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    """Build a Config from defaults, an optional YAML file and CLI ``key=value`` overrides."""
    cfg = OmegaConf.structured(Config)
    if path is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_object(cfg)


def save_config(cfg: Config, path: str) -> None:
    OmegaConf.save(OmegaConf.structured(cfg), path)
