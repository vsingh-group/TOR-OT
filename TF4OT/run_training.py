import functools
from TF4OT import tokenizers
from TF4OT import model
from TF4OT import utils
from datasets import load_dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import argparse
import logging
import json
import os

# Add DDP imports
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Add Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# Setup logging
logger = logging.getLogger(__name__)

from torch.utils.data._utils.collate import default_collate

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    torch.cuda.set_device(rank)
    dist.init_process_group( backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()

def prepare_dataset(cfg):
    dataset      = load_dataset(cfg.data.hf_dataset, cache_dir=cfg.data.hf_cache_dir)
    train_data   = dataset["train"]
    train_loader = DataLoader(train_data,
                            batch_size=cfg.training.batch_size_per_device,
                            shuffle=False, #Handled by DistributedSampler
                            num_workers=cfg.data.loading.num_workers,
                            pin_memory=True,
                            drop_last=True,
                            sampler=DistributedSampler(train_data),
                            collate_fn=utils.collate_from_string
    )
    len_train_data = len(train_data)
    return train_loader, len_train_data

def ddp_barrier():
    """Device-aware barrier that avoids the c10d warning, but stays version-safe."""
    if dist.is_available() and dist.is_initialized():
        try:
            # PyTorch >= 2.3 supports device_ids
            dist.barrier(device_ids=[torch.cuda.current_device()])
        except TypeError:
            # Older versions: no device_ids kwarg
            dist.barrier()

def training_routine(rank, cfg):
    """
    Train the regression model using Distributed Data Parallel (DDP).

    Args:
        rank: Rank of the current process
        world_size: Total number of processes
        cfg: Hydra configuration object
        model_save_path: Directory where to save the trained model (if None, uses default directory)
    """
    os.environ["HF_HOME"]            = cfg.data.hf_cache_dir
    os.environ["HF_DATASETS_CACHE"]  = cfg.data.hf_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cfg.data.hf_cache_dir
    os.environ["HF_METRICS_CACHE"]   = cfg.data.hf_cache_dir

    ddp_setup(rank, cfg.training.world_size)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        force=True)

    device = torch.device(
        getattr(cfg.training, "device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.backends.cudnn.benchmark = device.type == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True  # optional, faster on Ampere+
    use_amp = bool(getattr(cfg.training, "use_amp", True) and device.type == "cuda")

    # Create descriptive model name
    # if cfg.model.name is not defined then create a new name:
    name = cfg.model["name"] if "name" in cfg.model else None
    if name: model_name = f"{name}.pth"
    else: model_name = utils.get_model_name_from_cfg(cfg)

    # Set default directory if not provided
    os.makedirs(cfg.model.save_path, exist_ok=True)

    # Combine directory with auto-generated filename
    full_save_path = os.path.join(cfg.model.save_path, model_name)

    logger.info(f"Auto-generated model name: {model_name}")
    logger.info(f"Model will be saved to: {full_save_path}")

    # Loading training DATASET and loader
    logger.info("Loading dataset: " + cfg.data.hf_dataset)
    train_loader, len_train_data = prepare_dataset(cfg)

    logger.info(f"train_size={len_train_data}, training.batch_size={cfg.training.batch_size_per_device}, "
                f"batches_per_epoch={len(train_loader)}")

    # Create encoders
    logger.info("Creating Encoders")

    encoder_ab = nn.Sequential(nn.Linear(cfg.model.encoder.encoder_ab.input_dim, cfg.model.encoder.encoder_ab.emb_dim_ab),
                                nn.GELU(),
                                nn.Linear(cfg.model.encoder.encoder_ab.emb_dim_ab, cfg.model.encoder.encoder_ab.emb_dim_ab)
                                ).to(device)
    if not cfg.data.dataset.cost_fixed:
        encoder_c = nn.Sequential(nn.Linear(cfg.model.encoder.encoder_c.input_dim, cfg.model.encoder.encoder_c.emb_dim_c),
                                    nn.GELU(),
                                    nn.Linear(cfg.model.encoder.encoder_c.emb_dim_c, cfg.model.encoder.encoder_c.emb_dim_c)
                                    ).to(device)
    else: encoder_c = None

    # Create Tokenizer
    logger.info("Creating Tokenizer")
    if hasattr(cfg.tokenizer, "DigitTokenizer"):
        params = cfg.tokenizer.DigitTokenizer
        tokenizer = tokenizers.MyDigitTokenizer(
            vocabulary=params.vocabulary,
            decimal_precision=params.decimal_precision
        )
    elif hasattr(cfg.tokenizer, "IEEEFloatTokenizer"):
        params = cfg.tokenizer.IEEEFloatTokenizer
        tokenizer = tokenizers.IEEEFloatTokenizer(
            base=params.base,
            num_exponent_digits=params.num_exponent_digits,
            num_mantissa_digits=params.num_mantissa_digits
        )

    # Create Transformer Model
    logger.info("Creating Transformer Model")
    pytorch_model = model.TokenRegressor(tokenizer,
                                         encoder_ab=encoder_ab,
                                         encoder_c=encoder_c,
                                         emb_dim_ab=cfg.model.encoder.encoder_ab.emb_dim_ab,
                                         emb_dim_c=cfg.model.encoder.encoder_c.emb_dim_c,
                                         emb_dim_d=cfg.model.decoder.emb_dim_d,
                                         max_seq_len=cfg.model.decoder.max_seq_len,
                                         num_transformer_layers=cfg.model.decoder.num_transformer_layers,
                                         num_attention_heads=cfg.model.decoder.num_attention_heads,
                                         dropout=cfg.model.decoder.dropout
                                         ).to(device)
    pytorch_model.to(rank)
    pytorch_model = DDP(pytorch_model, device_ids=[rank])
    # the core model is now accessible as pytorch_model.module

    lr = cfg.training.learning_rate * torch.cuda.device_count()
    logger.info(f"Creating Optimizer with lr: {lr:.2e}")
    optimizer = optim.Adam(pytorch_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Print total model parameters
    model_total_params_trainable = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
    logger.info(f"Total model parameters: {model_total_params_trainable}")

    # Main training loop
    OT_INPUT_TOTAL        = "input_total"
    OT_INPUT_COST_MATRIX  = "input_cost_matrix"
    OT_TARGET_PLAN_MATRIX = "target_otplan_solve_sparse"

    logger.info("Starting training loop")
    pytorch_model.train()

    # Create a single progress bar for all epochs
    total_batches = cfg.training.num_epochs * len(train_loader)
    pbar = tqdm(total=total_batches, desc="Training")

    for epoch in range(cfg.training.num_epochs):
        train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:

            # encoder side
            encoder_input_ab = batch[OT_INPUT_TOTAL].to(device)

            if not cfg.data.dataset.cost_fixed:
                encoder_input_c      = batch[OT_INPUT_COST_MATRIX].flatten(start_dim=1).to(device)
                encoder_input_c_flat = encoder_input_c.flatten(start_dim=1).to(device)

            # tokenizer side (assumes your tokenizer.encode returns torch tensors)
            text_input, pad_mask = tokenizer.to_tokens(batch[OT_TARGET_PLAN_MATRIX], drop_last=True)
            #text_input, pad_mask = tokenizer2.to_tokens(batch[ot_output_map], drop_last=True)
            pad_mask = torch.cat([torch.full((encoder_input_ab.shape[0], 1), False), pad_mask], dim=1)
            text_target, _ = tokenizer.to_tokens(batch[OT_TARGET_PLAN_MATRIX], drop_last=False)

            #sanity check: convert from_tokens
            #text_target_from_tokens = tokenizer.from_tokens(text_target)
            #text_input_from_tokens  = tokenizer.from_tokens(text_input)

            # ignore PAD in targets
            text_target[text_target == tokenizer.size] = -100

            text_input  = text_input.to(device)
            pad_mask    = pad_mask.to(device)
            text_target = text_target.to(device)

            text_pred = pytorch_model.forward(encoder_input_ab,
                                              text_input,
                                              encoder_input_c=None if cfg.data.dataset.cost_fixed else encoder_input_c_flat,
                                              pad_mask=pad_mask)
            loss = criterion(text_pred.permute(0, 2, 1), text_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch: {epoch+1}/{cfg.training.num_epochs}, Loss: {loss.item():.5f}")
            pbar.update(1)

    pbar.close()

    # Save the trained model and vocabulary (only on main process)
    ddp_barrier()
    if not dist.is_initialized() or dist.get_rank() == 0:
        model_full_path  = full_save_path + ".pth"
        config_yaml_path = full_save_path + ".yaml"   # use YAML (keeps ${hydra:...} intact)

        logger.info(f"Saving model to {full_save_path}")
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

        # Save the underlying module’s weights if wrapped in DDP
        state_dict = (pytorch_model.module.state_dict()
                    if isinstance(pytorch_model, torch.nn.parallel.DistributedDataParallel)
                    else pytorch_model.state_dict())
        torch.save(state_dict, model_full_path)

        # Save the config WITHOUT resolving ${hydra:...} to avoid the crash
        with open(config_yaml_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=False))

        # (Optional) report param counts robustly even when encoder_C is None
        decoder_total_params = sum(p.numel() for p in state_dict.values())
        encoder_total_params = sum(p.numel() for p in encoder_ab.parameters() if p.requires_grad)
        total_params = decoder_total_params + encoder_total_params
        logger.info( "Total trainable parameters (encoder + decoder): {:,d} ({:.2f}M)".format(total_params, total_params / 1e6))
        logger.info(f"Total size on disk: {os.path.getsize(model_full_path) / (1024 * 1024):.2f} MB")
        logger.info(f"Training completed and model saved!: {model_full_path}")
    ddp_barrier()

    cleanup()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main function using Hydra for configuration management.

    Usage examples:
    # Use default configuration
    python3 run_training.py

    # Override specific parameters
    python3 run_training.py training.num_epochs=50 training.batch_size=8
    """
    #torch.manual_seed(123)

    logger.info("Using GPU training mode")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")

    # Set world_size if not specified
    if cfg.training.world_size is None: cfg.training.world_size = torch.cuda.device_count()

    mp.spawn(training_routine, args=(cfg, ), nprocs=cfg.training.world_size)

if __name__ == "__main__":
    main()