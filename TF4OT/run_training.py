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

# Add distributed training imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Add Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# Setup logging
logger = logging.getLogger(__name__)

from torch.utils.data._utils.collate import default_collate

def training_routine(cfg):
    """
    Train the regression model using Distributed Data Parallel (DDP).

    Args:
        rank: Rank of the current process
        world_size: Total number of processes
        cfg: Hydra configuration object
        model_save_path: Directory where to save the trained model (if None, uses default directory)
    """
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
    else:
        model_name = (
            f"model_n{cfg.data.dataset.size}"
            f"_support{cfg.data.dataset.support_size}"
            f"_ep{cfg.training.num_epochs}"
            f"_dec_units_{cfg.model.decoder.units}"
            f"_nlayers_{cfg.model.decoder.num_transformer_layers}"
            f"_nheads_{cfg.model.decoder.num_attention_heads}"
            f"_drop_{cfg.model.decoder.dropout}"
        )

    # Set default directory if not provided
    os.makedirs(cfg.model.save_path, exist_ok=True)

    # Combine directory with auto-generated filename
    full_save_path = os.path.join(cfg.model.save_path, model_name)

    logger.info(f"Auto-generated model name: {model_name}")
    logger.info(f"Model will be saved to: {full_save_path}")

    # Loading training dataset and loader
    logger.info("Loading dataset: " + cfg.data.hf_dataset)
    dataset      = load_dataset(cfg.data.hf_dataset, cache_dir=cfg.data.hf_cache_dir)
    train_data   = dataset["train"]
    train_loader = DataLoader(train_data,
                            batch_size=cfg.training.batch_size,
                            shuffle=False,
                            num_workers=cfg.data.loading.num_workers,
                            pin_memory=(device.type == "cuda"),
                            collate_fn=utils.collate_from_string
    )
    logger.info(f"train_size={len(train_data)}, training.batch_size={cfg.training.batch_size}, "
                f"batches_per_epoch={len(train_loader)}")

    # Create encoder model
    logger.info("Creating Encoder")
    encoder_ab = nn.Sequential(nn.Linear(cfg.data.dataset.input_total_dim, cfg.model.encoder.emb_dim),
                               nn.GELU(),
                               nn.Linear(cfg.model.encoder.emb_dim, cfg.model.encoder.emb_dim)
                               ).to(device)
    encoder_C = nn.Sequential(nn.Linear(cfg.data.dataset.input_cost_matrix_dim, cfg.model.encoder.emb_dim),
                              nn.GELU(),
                              nn.Linear(cfg.model.encoder.emb_dim, cfg.model.encoder.emb_dim)
                              ).to(device)

    # Create Tokenizer
    logger.info("Creating Tokenizer")
    # TODO: add support for other tokenizers
    # tokenizer = tokenizers.MyDigitTokenizer(vocabulary=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', '.', '[', ']', ' '],
    #                                         decimal_precision=cfg.tokenizer.decimal_precision)
    tokenizer = tokenizers.IEEEFloatTokenizer(base=cfg.tokenizer.base,
                                               num_exponent_digits=cfg.tokenizer.num_exponent_digits,
                                               num_mantissa_digits=cfg.tokenizer.num_mantissa_digits)

    # Create Transformer Model
    logger.info("Creating Transformer Model")
    pytorch_model = model.TokenRegressor(tokenizer,
                                         encoder_ab,
                                         encoder_C,
                                         cfg.model.decoder.emb_dim,
                                         cfg.model.decoder.seq_dim,
                                         cfg.model.decoder.num_transformer_layers,
                                         cfg.model.decoder.num_attention_heads,
                                         cfg.model.decoder.dropout
                                         ).to(device)

    optimizer = optim.Adam(pytorch_model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Print total model parameters
    model_total_params = sum(p.numel() for p in pytorch_model.parameters())
    model_total_params_trainable = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
    logger.info(f"Total model parameters: {model_total_params}")

    # Main training loop
    OT_INPUT_TOTAL        = "input_total"
    OT_INPUT_COST_MATRIX  = "input_cost_matrix"
    OT_TARGET_PLAN_MATRIX = "target_otplan_solve_sparse"

    logger.info("Starting training loop")
    pytorch_model.train()
    for epoch in range(cfg.training.num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        for batch in pbar:

            # encoder side
            encoder_input_ab = batch[OT_INPUT_TOTAL]
            encoder_input_ab = encoder_input_ab.to(device)

            encoder_input_C  = batch[OT_INPUT_COST_MATRIX]
            encoder_input_C_flat = encoder_input_C.flatten(start_dim=1)
            encoder_input_C_flat = encoder_input_C_flat.to(device)

            # tokenizer side (assumes your tokenizer.encode returns torch tensors)
            text_input, pad_mask = tokenizer.to_tokens(batch[OT_TARGET_PLAN_MATRIX], drop_last=True)
            #text_input, pad_mask = tokenizer2.to_tokens(batch[ot_output_map], drop_last=True)
            pad_mask = torch.cat([torch.full((encoder_input_ab.shape[0], 1), False), pad_mask], dim=1)
            text_target, _ = tokenizer.to_tokens(batch[OT_TARGET_PLAN_MATRIX], drop_last=False)

            #sanity check: convert from_tokens
            text_target_from_tokens = tokenizer.from_tokens(text_target)
            text_input_from_tokens  = tokenizer.from_tokens(text_input)


            # ignore PAD in targets
            text_target[text_target == tokenizer.pad_id] = -100

            text_input  = text_input.to(device)
            pad_mask    = pad_mask.to(device)
            text_target = text_target.to(device)

            text_pred = pytorch_model.forward(encoder_input_ab,
                                              text_input,
                                              encoder_input_C=None,
                                              pad_mask=pad_mask)
            loss = criterion(text_pred.permute(0, 2, 1), text_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch: {epoch+1}, Batch Loss: {loss.item():.5}")

    # # Wrap model with DDP only if not using CPU
    # if not use_cpu: decoder = DDP(decoder, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # if use_cpu:
    #     # For CPU training, use regular DataLoader
    #     train_loader = DataLoader(
    #         train_dataset,
    #         batch_size=cfg.training.batch_size,
    #         #shuffle=True,
    #         num_workers=cfg.data.loading.num_workers,
    #         pin_memory=False,  # No pin_memory for CPU
    #     )
    # else:
    #     # Use DistributedSampler for proper data distribution across GPUs
    #     train_sampler = DistributedSampler(
    #         train_dataset,
    #         num_replicas=world_size,
    #         rank=rank,
    #         #shuffle=True
    #     )

    #     train_loader = DataLoader(
    #         train_dataset,
    #         batch_size=cfg.training.batch_size,
    #         sampler=train_sampler,
    #         num_workers=cfg.data.loading.num_workers,
    #         pin_memory=True
    #     )

    rank = 0
    # Save the trained model and vocabulary (only on main process)
    if rank == 0:
        model_full_path  = full_save_path + ".pth"
        config_full_path = full_save_path + ".json"

        logger.info(f"Saving model to {full_save_path}")
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
        torch.save(pytorch_model.state_dict(), model_full_path)

        # Save the full config automatically using OmegaConf
        # to_container() converts DictConfig to a standard Python dict
        config = OmegaConf.to_container(cfg, resolve=True)
        with open(config_full_path, "w") as f:
            json.dump(config, f, indent=2)


        # TODO: all params included in decoder + encoder 
        decoder_total_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
        encoder_total_params = sum(p.numel() for p in encoder_C.parameters() if p.requires_grad) + sum(p.numel() for p in encoder_ab.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters (encoder + decoder): {decoder_total_params + encoder_total_params}")
        logger.info(f"Total size on disk: {os.path.getsize(model_full_path) / (1024 * 1024):.2f} MB")
        logger.info(f"Training completed and model saved!: {model_full_path}")

    # # Cleanup distributed training
    # cleanup_distributed(use_cpu)
    # return (decoder.module if not use_cpu else decoder), vocab

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

    logger.info("Using GPU training mode")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")

    # Set world_size if not specified
    if cfg.training.world_size is None: cfg.training.world_size = torch.cuda.device_count()

    #TODO: logger.info(f"Starting distributed training with {cfg.training.world_size} GPUs")
    # For GPU training, run directly without multiprocessing
    training_routine(cfg)

if __name__ == "__main__":
    main()