"""Training entry point.

Single GPU:
    python -m torot.train --config torot/configs/default.yaml run.out_dir=experiments/torot/base

Multi-GPU (DDP):
    torchrun --nproc_per_node 4 -m torot.train --config torot/configs/default.yaml

Any config field can be overridden with ``section.key=value``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from torot.config import Config, load_config, save_config
from torot.data.dataset import Collator, OTPlanDataset, SizeBucketBatchSampler, batch_to_device
from torot.decode import predict_plans
from torot.loss import PlanLoss, token_accuracy
from torot.metrics import aggregate, evaluate_instance
from torot.model import ToROT
from torot.tokenizer import PlanTokenizer


# --------------------------------------------------------------------------- distributed helpers
def ddp_init() -> tuple[int, int, torch.device]:
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return rank, world, torch.device("cuda", local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def unwrap(model: torch.nn.Module) -> ToROT:
    return model.module if isinstance(model, DDP) else model


# --------------------------------------------------------------------------- optimisation
def build_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW with weight decay on matrices only (no decay for norms, biases, embeddings)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 and "emb" not in name else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.optim.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.optim.lr, betas=tuple(cfg.optim.betas), fused=torch.cuda.is_available())


def lr_at(step: int, cfg: Config) -> float:
    """Linear warmup then cosine decay to ``min_lr_ratio * lr``."""
    o = cfg.optim
    if step < o.warmup_steps:
        return o.lr * (step + 1) / o.warmup_steps
    progress = min(1.0, (step - o.warmup_steps) / max(1, o.max_steps - o.warmup_steps))
    return o.lr * (o.min_lr_ratio + (1.0 - o.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


# --------------------------------------------------------------------------- validation
@torch.no_grad()
def validate(
    model: ToROT,
    loss_fn: PlanLoss,
    tok: PlanTokenizer,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    accs: List[Dict[str, float]] = []
    decoded: List[Dict[str, float]] = []
    n_decoded = 0
    for i, batch in enumerate(loader):
        if i >= cfg.run.val_batches:
            break
        batch = batch_to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.run.bf16 and device.type == "cuda"):
            logits = model(batch)
        losses.append(loss_fn(logits, batch["target_ids"], batch["kinds"]).item())
        accs.append(token_accuracy(logits, batch["target_ids"], batch["kinds"]))
        if n_decoded < cfg.run.val_decode_instances:
            instances = batch["instances"][: cfg.run.val_decode_instances - n_decoded]
            plans = predict_plans(model, tok, instances, device, max_len=cfg.run.max_decode_len)
            decoded += [evaluate_instance(P, inst) for P, inst in zip(plans, instances)]
            n_decoded += len(instances)
    out = {"val/loss": sum(losses) / max(1, len(losses))}
    for k in {k for a in accs for k in a}:
        out[f"val/{k}"] = sum(a[k] for a in accs if k in a) / max(1, sum(k in a for a in accs))
    out.update({f"val/{k}": v for k, v in aggregate(decoded).items()})
    model.train()
    return out


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(path: str, model: ToROT, ema: AveragedModel, optimizer, step: int, best: float, cfg: Config) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val": best,
        },
        path,
    )
    save_config(cfg, os.path.join(os.path.dirname(path), "config.yaml"))


# --------------------------------------------------------------------------- main
def train(cfg: Config) -> None:
    rank, world, device = ddp_init()
    torch.manual_seed(cfg.run.seed + rank)
    if is_main():
        os.makedirs(cfg.run.out_dir, exist_ok=True)
        save_config(cfg, os.path.join(cfg.run.out_dir, "config.yaml"))

    tok = PlanTokenizer(cfg.tokenizer.mantissa_digits, cfg.tokenizer.max_exponent)
    collate = Collator(tok)

    ds = load_dataset(cfg.data.repo_id, cache_dir=cfg.data.cache_dir)
    train_ds = OTPlanDataset(ds[cfg.data.train_split], cfg.data.transpose_aug_prob)
    val_ds = OTPlanDataset(ds[cfg.data.val_split])
    train_sampler = SizeBucketBatchSampler(
        train_ds.sizes(), cfg.optim.batch_size, shuffle=True, seed=cfg.run.seed, rank=rank, world_size=world
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=collate, num_workers=cfg.data.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=SizeBucketBatchSampler(val_ds.sizes(), cfg.optim.batch_size, shuffle=False),
        collate_fn=collate,
        num_workers=cfg.data.num_workers,
    )

    base_model = ToROT(cfg.model, tok.vocab_size).to(device)
    if is_main():
        print(f"parameters: {base_model.num_parameters() / 1e6:.1f}M, train instances: {len(train_ds)}")
    ema = AveragedModel(base_model, multi_avg_fn=get_ema_multi_avg_fn(cfg.optim.ema_decay), use_buffers=True)
    model: torch.nn.Module = DDP(base_model, device_ids=[device.index]) if world > 1 else base_model
    if cfg.run.compile:
        model = torch.compile(model)
    loss_fn = PlanLoss(tok, cfg.loss.digit_weight_decay, cfg.loss.exponent_weight, cfg.loss.ordinal_eps).to(device)
    optimizer = build_optimizer(base_model, cfg)

    step, best_val = 0, float("inf")
    if cfg.run.resume:
        ckpt = torch.load(cfg.run.resume, map_location=device)
        base_model.load_state_dict(ckpt["model"])
        ema.module.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, best_val = ckpt["step"], ckpt["best_val"]

    log_path = os.path.join(cfg.run.out_dir, "log.jsonl")
    model.train()
    epoch = 0
    t0 = time.time()
    while step < cfg.optim.max_steps:
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if step >= cfg.optim.max_steps:
                break
            batch = batch_to_device(batch, device)
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, cfg)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.run.bf16 and device.type == "cuda"):
                logits = model(batch)
            loss = loss_fn(logits, batch["target_ids"], batch["kinds"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), cfg.optim.grad_clip)
            optimizer.step()
            ema.update_parameters(base_model)
            step += 1

            if is_main() and step % cfg.run.log_every == 0:
                rec = {
                    "step": step,
                    "loss": loss.item(),
                    "lr": lr_at(step, cfg),
                    "grad_norm": float(grad_norm),
                    "tokens": int(batch["kinds"].ge(0).sum()),
                    "elapsed_s": time.time() - t0,
                    **token_accuracy(logits, batch["target_ids"], batch["kinds"]),
                }
                print(json.dumps(rec))
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

            if step % cfg.run.val_every == 0 or step == cfg.optim.max_steps:
                if is_main():
                    metrics = validate(ema.module, loss_fn, tok, val_loader, cfg, device)
                    metrics["step"] = step
                    print(json.dumps(metrics))
                    with open(log_path, "a") as f:
                        f.write(json.dumps(metrics) + "\n")
                    if metrics["val/loss"] < best_val:
                        best_val = metrics["val/loss"]
                        save_checkpoint(os.path.join(cfg.run.out_dir, "best.pt"), base_model, ema, optimizer, step, best_val, cfg)
                    save_checkpoint(os.path.join(cfg.run.out_dir, "last.pt"), base_model, ema, optimizer, step, best_val, cfg)
                if dist.is_initialized():
                    dist.barrier()
        epoch += 1

    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML config file")
    p.add_argument("overrides", nargs="*", help="section.key=value overrides")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
