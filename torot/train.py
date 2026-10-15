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

# variable-length batches fragment the CUDA allocator; expandable segments avoid spurious OOMs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from typing import Dict, List, Optional

import copy

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from torot.config import Config, load_config, save_config
from torot.data.dataset import (
    Collator,
    OTPlanDataset,
    OTRowDataset,
    SizeBucketBatchSampler,
    batch_to_device,
    cost_label_map,
    load_corpus,
    record_to_instance,
)
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


# --------------------------------------------------------------------------- EMA
class EMA:
    """Exponential moving average of the weights, evaluated and checkpointed as ``best.pt``.

    The decay is warmed up, ``min(decay, (1 + t) / (10 + t))``, so that early in training
    the average tracks the live weights instead of the random initialisation.
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for p_ema, p in zip(self.module.parameters(), model.parameters()):
            p_ema.lerp_(p.detach(), 1.0 - d)

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state["module"])
        self.updates = state["updates"]


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
    decode_instances: List[dict],
    cfg: Config,
    device: torch.device,
    decode_batch: int = 64,
) -> Dict[str, float]:
    """Teacher-forced loss/accuracy on ``val_batches`` batches, plus metrics of plans decoded
    for a fixed list of whole validation instances (the same list at every validation)."""
    model.eval()
    losses: List[float] = []
    accs: List[Dict[str, float]] = []
    decoded: List[Dict[str, float]] = []
    for start in range(0, len(decode_instances), decode_batch):
        chunk = decode_instances[start : start + decode_batch]
        plans = predict_plans(model, tok, chunk, device, max_len=cfg.run.max_decode_len)
        decoded += [evaluate_instance(P, inst) for P, inst in zip(plans, chunk)]
    for i, batch in enumerate(loader):
        if i >= cfg.run.val_batches:
            break
        batch = batch_to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.run.bf16 and device.type == "cuda"):
            logits = model(batch)
        losses.append(loss_fn(logits, batch["target_ids"], batch["kinds"]).item())
        accs.append(token_accuracy(logits, batch["target_ids"], batch["kinds"]))
    out = {"val/loss": sum(losses) / max(1, len(losses))}
    for k in {k for a in accs for k in a}:
        out[f"val/{k}"] = sum(a[k] for a in accs if k in a) / max(1, sum(k in a for a in accs))
    out.update({f"val/{k}": v for k, v in aggregate(decoded).items()})
    model.train()
    return out


# --------------------------------------------------------------------------- progress file
def write_progress(
    path: str, cfg: Config, step: int, start_step: int, t0: float, train_loss: float, last_val: Dict[str, float], best_val: float
) -> None:
    """Overwrite a short human-readable status file (``progress.txt``) for quick checks."""
    elapsed = time.time() - t0
    rate = elapsed / max(1, step - start_step)
    remaining = (cfg.optim.max_steps - step) * rate
    lines = [
        f"updated     {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"step        {step} / {cfg.optim.max_steps}  ({100.0 * step / cfg.optim.max_steps:.1f}%)",
        f"elapsed     {elapsed / 3600:.2f} h   ETA {remaining / 3600:.2f} h   ({rate:.3f} s/step)",
        f"train loss  {train_loss:.4f}   lr {lr_at(step, cfg):.2e}",
    ]
    if last_val:
        lines.append(
            "last val    step {step}  loss {loss:.4f}  ptr_acc {ptr:.3f}  exp_acc {exp:.3f}  digit1_acc {d1:.3f}".format(
                step=last_val.get("step", 0),
                loss=last_val.get("val/loss", float("nan")),
                ptr=last_val.get("val/acc/tgt_ptr", float("nan")),
                exp=last_val.get("val/acc/exponent", float("nan")),
                d1=last_val.get("val/acc/digit1", float("nan")),
            )
        )
        lines.append(
            "decoded     mape_med {mape:.4f}  cost_gap_med {gap:+.2e}  max_viol_med {viol:.2e}  support_recall {rec:.3f}".format(
                mape=last_val.get("val/mape_ref_support/median", float("nan")),
                gap=last_val.get("val/cost_gap/median", float("nan")),
                viol=last_val.get("val/max_violation/median", float("nan")),
                rec=last_val.get("val/support_recall/mean", float("nan")),
            )
        )
    lines.append(f"best val    loss {best_val:.4f}" if best_val < float("inf") else "best val    (none yet)")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(
    path: str, model: ToROT, ema: EMA, optimizer, step: int, epoch: int, best: float, cfg: Config
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.module.state_dict(),
            "ema_updates": ema.updates,
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
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

    ds = load_corpus(cfg.data.repo_id, cfg.data.cache_dir)
    if cfg.model.decoding == "row":
        train_ds = OTRowDataset(ds[cfg.data.train_split], cfg.data.transpose_aug_prob, cfg.data.residual_noise)
        val_ds = OTRowDataset(ds[cfg.data.val_split])
    else:
        train_ds = OTPlanDataset(ds[cfg.data.train_split], cfg.data.transpose_aug_prob)
        val_ds = OTPlanDataset(ds[cfg.data.val_split])
    val_split = ds[cfg.data.val_split]
    decode_idx = np.random.default_rng(cfg.run.seed).permutation(len(val_split))[: cfg.run.val_decode_instances]
    val_cost_map = cost_label_map(val_split)
    decode_instances = [record_to_instance(val_split[int(k)], val_cost_map) for k in sorted(decode_idx)]
    train_sampler = SizeBucketBatchSampler(
        train_ds.sizes(),
        cfg.optim.batch_size,
        max_tokens=cfg.optim.max_tokens,
        shuffle=True,
        seed=cfg.run.seed,
        rank=rank,
        world_size=world,
        tokens_per_bin=train_ds.tokens_per_size,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=collate, num_workers=cfg.data.num_workers, pin_memory=True
    )
    # shuffled with a fixed seed: a deterministic validation subset that mixes support sizes
    val_loader = DataLoader(
        val_ds,
        batch_sampler=SizeBucketBatchSampler(
            val_ds.sizes(),
            cfg.optim.batch_size,
            max_tokens=cfg.optim.max_tokens,
            shuffle=True,
            seed=cfg.run.seed,
            tokens_per_bin=val_ds.tokens_per_size,
        ),
        collate_fn=collate,
        num_workers=cfg.data.num_workers,
    )

    base_model = ToROT(cfg.model, tok.vocab_size).to(device)
    if is_main():
        unit = "row examples" if cfg.model.decoding == "row" else "instances"
        print(f"parameters: {base_model.num_parameters() / 1e6:.1f}M, decoding: {cfg.model.decoding}, train {unit}: {len(train_ds)}")
    ema = EMA(base_model, cfg.optim.ema_decay)
    model: torch.nn.Module = DDP(base_model, device_ids=[device.index]) if world > 1 else base_model
    if cfg.run.compile:
        model = torch.compile(model)
    loss_fn = PlanLoss(
        tok, cfg.loss.digit_weight_decay, cfg.loss.exponent_weight, cfg.loss.ordinal_eps, cfg.loss.z_loss
    ).to(device)
    optimizer = build_optimizer(base_model, cfg)

    step, epoch, best_val = 0, 0, float("inf")
    if cfg.run.resume:
        ckpt = torch.load(cfg.run.resume, map_location=device)
        base_model.load_state_dict(ckpt["model"])
        ema.load_state_dict({"module": ckpt["ema"], "updates": ckpt.get("ema_updates", ckpt["step"])})
        optimizer.load_state_dict(ckpt["optimizer"])
        step, best_val = ckpt["step"], ckpt["best_val"]
        epoch = ckpt.get("epoch", 0) + 1          # restart from a fresh data order, not the saved epoch's
        if is_main():
            print(f"resumed from {cfg.run.resume} at step {step} (epoch {epoch})")

    log_path = os.path.join(cfg.run.out_dir, "log.jsonl")
    progress_path = os.path.join(cfg.run.out_dir, "progress.txt")
    start_step = step
    last_val: Dict[str, float] = {}
    model.train()
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
            ema.update(base_model)
            step += 1

            if is_main() and step % cfg.run.log_every == 0:
                rec = {
                    "step": step,
                    "loss": loss.item(),
                    "lr": lr_at(step, cfg),
                    "grad_norm": float(grad_norm),
                    "tokens": int(batch["kinds"].ge(0).sum()),
                    "elapsed_s": time.time() - t0,
                    "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
                    **token_accuracy(logits, batch["target_ids"], batch["kinds"]),
                }
                print(json.dumps(rec), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                write_progress(progress_path, cfg, step, start_step, t0, loss.item(), last_val, best_val)

            if step % cfg.run.val_every == 0 or step == cfg.optim.max_steps:
                if is_main():
                    metrics = validate(ema.module, loss_fn, tok, val_loader, decode_instances, cfg, device)
                    metrics["step"] = step
                    print(json.dumps(metrics), flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps(metrics) + "\n")
                    if metrics["val/loss"] < best_val:
                        best_val = metrics["val/loss"]
                        save_checkpoint(os.path.join(cfg.run.out_dir, "best.pt"), base_model, ema, optimizer, step, epoch, best_val, cfg)
                    save_checkpoint(os.path.join(cfg.run.out_dir, "last.pt"), base_model, ema, optimizer, step, epoch, best_val, cfg)
                    last_val = metrics
                    write_progress(progress_path, cfg, step, start_step, t0, loss.item(), last_val, best_val)
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
