import functools
from TF4OT import tokenizers
from TF4OT import model
#from TF4OT.serialization.numeric import tokens
#from TF4OT.multitarget_models import *
#from TF4OT.datasets import load_random_uniform_data_1d, load_ot_data_single_target_format
#rom TF4OT.utils.multitarget_datasets_hf import *
from datasets import load_dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import logging
import json
from tqdm import tqdm

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
            f"model_n{cfg.data.dataset.size_s}"
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
    dataset = load_dataset(cfg.data.hf_dataset, cache_dir=cfg.data.hf_cache_dir)
    train_data = dataset["train"]
    train_loader = DataLoader(train_data, 
                            batch_size=cfg.training.batch_size,
                            shuffle=True,
                            num_workers=cfg.data.loading.num_workers,
                            pin_memory=True if device.type == "cuda" else False)

    logger.info(f"train_size={len(train_data)}, training.batch_size={cfg.training.batch_size}, "
                f"batches_per_epoch={len(train_loader)}")

    # Create encoder model
    logger.info("Creating Encoder")
    encoder = nn.Sequential(nn.Linear(1 + 2 * cfg.data.dataset.support_size, cfg.model.encoder.emb_dim),
                            nn.GELU(),
                            nn.Linear(cfg.model.encoder.emb_dim, cfg.model.encoder.emb_dim)
                            ).to(device)

    # Create Tokenizer
    logger.info("Creating Tokenizer")
    # TODO: add support for other tokenizers
    tokenizer = tokenizers.MyDigitTokenizer(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', '.', '[', ']', ' '])

    tokenizer2 = tokenizers.DigitTokenizer(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', '.', '[', ']', ' '])

    # Create Transformer Model
    logger.info("Creating Model")
    pytorch_model = model.TokenRegressor(tokenizer,
                                        encoder,
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
    ot_output       = "output_plan_C"
    ot_output_map   = "output_map_sparse"
    ot_input_ab     = "total_input"
    ot_input_C      = "input_C"

    logger.info("Starting training loop")
    pytorch_model.train()
    for epoch in range(cfg.training.num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        for batch in pbar:

            # encoder side
            encoder_input = torch.stack(batch[ot_input_ab], dim=1).to(torch.float32)
            encoder_input = encoder_input.to(device)

            # tokenizer side (assumes your tokenizer.encode returns torch tensors)
            text_input, pad_mask = tokenizer.encode(batch[ot_output_map], drop_last=True)
            text_input, pad_mask = tokenizer2.to_tokens(batch[ot_output_map], drop_last=True)
            pad_mask = torch.cat([torch.full((encoder_input.shape[0], 1), False), pad_mask], dim=1)
            text_target, _ = tokenizer.encode(batch[ot_output_map], drop_last=False)

            # ignore PAD in targets
            text_target[text_target == tokenizer.size] = -100

            text_input = text_input.to(device)
            pad_mask = pad_mask.to(device)
            text_target = text_target.to(device)

            text_pred = pytorch_model.forward(encoder_input, text_input, pad_mask)
            loss = criterion(text_pred.permute(0, 2, 1), text_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch: {epoch+1}, Batch Loss: {loss.item():.5}")

    # Define encoder
    # encoder = SimpleEncoderMultitarget(support_size=cfg.model.encoder.support_size,
    #                                    input_dim=cfg.model.encoder.input_dim,
    #                                    output_dim=cfg.model.encoder.output_dim).to(device)

    # Create a vocabulary for the model
    # Use a unnormalized vocabulary for regression.
    # specify number of digits for mantissa and exponent
    # Build a serializer with custom defaults:
    # if rank == 0: logger.info("Creating vocabulary")
    # my_serializer = tokens.IEEEFloatTokenSerializer(
    #     num_exponent_digits=cfg.tokenizer.num_exponent_digits,
    #     num_mantissa_digits=cfg.tokenizer.num_mantissa_digits
    # )
    # vocab = vocabs.UnnormalizedVocab(serializer=my_serializer)
    # if rank == 0:
    #     logger.info(f"vocab.size (V): {vocab.size}")
    #     logger.info(f"vocab.token_length (K): {vocab.token_length}") # K = 2 + num_exponent + num_mantissa

    # # Create decoder model
    # if rank == 0: logger.info("Creating decoder model")
    # decoder = AttentionDecoderMultitarget(
    #     encoder=encoder,
    #     vocab=vocab,
    #     units=cfg.model.decoder.units,
    #     num_transformer_layers=cfg.model.decoder.num_transformer_layers,
    #     num_attention_heads=cfg.model.decoder.num_attention_heads,
    #     num_targets=cfg.data.dataset.num_targets,
    #     dropout=cfg.model.decoder.dropout,
    # ).to(device)

    # # Initialize model parameters with a dummy forward pass
    # # This is required for DDP to work properly
    # with torch.no_grad():
    #     B = 2
    #     K = cfg.data.dataset.num_targets  # number of targets for multitarget decoder
    #     supp = cfg.model.encoder.support_size
    #     # Encoder-side dummy inputs (match training shapes & dtypes)
    #     dummy_u  = torch.rand(B, supp, device=device, dtype=torch.float32)  # [B, support_size]
    #     dummy_v  = torch.rand(B, supp, device=device, dtype=torch.float32)  # [B, support_size]
    #     dummy_ux = torch.arange(1, supp + 1, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)  # [B, support_size] (each row: 1 to support_size)
    #     # token_length is how many tokens represent one float; model uses it for pos emb.
    #     L_tok = getattr(vocab, "token_length", vocab.token_length)  # safe fallback if not set
    #     dummy_dec_in = torch.randint(
    #         low=0, high=vocab.size, size=(B, K, L_tok - 1), device=device, dtype=torch.long
    #     )  # [B, K, L-1] for multitarget decoder
    #     # Run a dummy forward to materialize LazyLinear etc.
    #     _ = decoder((dummy_u, dummy_v, dummy_ux), dummy_dec_in)  # logits [B, K, L, V]

    # # Wrap model with DDP only if not using CPU
    # if not use_cpu: decoder = DDP(decoder, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # # Load test data
    # if rank == 0: logger.info(f"Loading data from: {cfg.data.data_hf}")

    # # loading dataset - huggingface
    # train_dataset = load_dataset(cfg.data.data_hf, split="train")
    # dataset_size  = len(train_dataset)
    # X_cols = ["u_source", "v_target", "u_source_support"]  # features
    # Y_col  = "T_pi"                                        # target (barycentric projection)

    # if rank == 0: logger.info(f"Loaded {dataset_size} training samples")

    # # Convert Y to token IDs -- faster
    # def encode_batch(batch):
    #     # batch[Y_col]: list of rows, each row is list[float]
    #     return {
    #         "Y_token_ids": [[vocab.to_int(y) for y in row] for row in batch[Y_col]]
    #     }

    # train_dataset = train_dataset.map(
    #     encode_batch,
    #     batched=True,
    #     batch_size=cfg.training.batch_size,
    #     num_proc=1,          # parallelize
    #     desc="Tokenizing Y"
    # )

    # # # --- Build tuple-like fields: X, Y_in, Y_out ---
    # # def build_fields(example):
    # #     # Convert feature blocks to arrays (float32) and concat along the last axis
    # #     # Assumes each X_cols entry is array-like with matching leading dims
    # #     x_parts = [np.asarray(example[c], dtype=np.float32) for c in X_cols]
    # #     # If these are 2D/3D blocks, concat along the last axis; adapt axis if needed
    # #     X = np.concatenate(x_parts, axis=-1)

    # #     # Y: convert to int64 tensor-friendly array
    # #     Y = np.asarray(example["Y_token_ids"], dtype=np.int64)
    # #     # Teacher-forcing style split: shift last axis by 1
    # #     Y_in  = Y[..., :-1]
    # #     Y_out = Y  # full target

    # #     return {
    # #         "X": X,
    # #         "Y_in": Y_in,
    # #         "Y_out": Y_out,
    # #     }
    # # train_dataset = train_dataset.map(build_fields, desc="Building X, Y_in, Y_out")

    # # Keep only the three fields we need
    # keep_cols = ["u_source", "v_target", "u_source_support", "T_pi", "pi", "Y_token_ids"]
    # drop_cols = [c for c in train_dataset.column_names if c not in keep_cols]
    # if drop_cols: train_dataset = train_dataset.remove_columns(drop_cols)
    # # Format as PyTorch so __getitem__ returns tensors
    # train_dataset.set_format(type="torch", columns=keep_cols)

    # # Create dataset and dataloader
    # #train_dataset = TensorDataset(X_train_t, Y_train_token_ids_t[:, :, :-1], Y_train_token_ids_t)

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

    # # Setup optimizer and loss function
    # optimizer = optim.Adam(decoder.parameters(), lr=cfg.training.learning_rate)
    # criterion = nn.CrossEntropyLoss(reduction='sum')
    # scaler    = torch.amp.GradScaler() if cfg.training.use_amp else None

    # # Training loop
    # if rank == 0:
    #     logger.info("Starting training loop")
    #     logger.info(f"Training for {cfg.training.num_epochs} epochs with batch size {cfg.training.batch_size}")
    #     logger.info(f"Learning rate: {cfg.training.learning_rate}")
    #     logger.info(f"Using AMP: {cfg.training.use_amp}")
    #     logger.info(f"Gradient accumulation: {cfg.training.grad_accumulation}")

    # decoder.train()

    # # Create epoch progress bar (only on main process)
    # if rank == 0:
    #     epoch_pbar = tqdm(range(cfg.training.num_epochs), desc="Training", unit="epoch")
    # else:
    #     epoch_pbar = range(cfg.training.num_epochs)

    # for epoch in epoch_pbar:
    #     # Set epoch for sampler to ensure different shuffling each epoch (only for distributed training)
    #     if not use_cpu: train_sampler.set_epoch(epoch)

    #     total_loss = 0.0
    #     optimizer.zero_grad()

    #     # Create batch progress bar (only on main process)
    #     #TODO: fix for multi-gpu training
    #     if rank == 0:
    #         batch_pbar = train_loader
    #     else:
    #         batch_pbar = train_loader

    #     for step, batch in enumerate(batch_pbar):
    #         batch_X_u             = batch['u_source']           # x_1
    #         batch_X_v             = batch['v_target']           # x_2
    #         batch_X_ux            = batch['u_source_support']   # x_3
    #         batch_Y_map           = batch['T_pi']               # y
    #         batch_Y_map_token_ids = batch['Y_token_ids']        # y_target: [B, K, L]
    #         batch_Y_plan          = batch['pi']

    #         batch_X_u             = batch_X_u.to(device, non_blocking=True)
    #         batch_X_v             = batch_X_v.to(device, non_blocking=True)
    #         batch_X_ux            = batch_X_ux.to(device, dtype=torch.long, non_blocking=True)
    #         batch_Y_map           = batch_Y_map.to(device, non_blocking=True)
    #         batch_Y_map_token_ids = batch_Y_map_token_ids.to(device, non_blocking=True)
    #         batch_Y_plan          = batch_Y_plan.to(device, non_blocking=True)

    #         # Forward pass
    #         #TODO: Enable quantization with cfg.training.use_amp:
    #         batch_logits = decoder((batch_X_u, batch_X_v, batch_X_ux), batch_Y_map_token_ids[:,:,:-1])  # [B, K, L, V]
    #         #TODO: adding .contiguous() might be faster
    #         logits_ch_first = batch_logits.permute(0, 3, 1, 2)                                          # [B, V, K, L]
    #         loss = criterion(logits_ch_first, batch_Y_map_token_ids)                                    # averages over B,K,L

    #         # Backward pass with gradient accumulation and AMP support
    #         if cfg.training.use_amp:
    #             scaler.scale(loss / cfg.training.grad_accumulation).backward()
    #         else:
    #             (loss / cfg.training.grad_accumulation).backward()

    #         if (step + 1) % cfg.training.grad_accumulation == 0 or (step + 1) == len(train_loader):
    #             if cfg.training.use_amp:
    #                 scaler.step(optimizer)
    #                 scaler.update()
    #             else:
    #                 optimizer.step()
    #             optimizer.zero_grad()

    #         total_loss += loss.item()

    #     # Synchronize loss across all processes (only for distributed training)
    #     if use_cpu:
    #         avg_loss = total_loss / len(train_loader)
    #     else:
    #         total_loss_tensor = torch.tensor(total_loss, device=device)
    #         dist.all_reduce(total_loss_tensor)
    #         avg_loss = total_loss_tensor.item() / world_size / len(train_loader)

    #     # Update epoch progress bar with final loss
    #     if rank == 0:
    #         epoch_pbar.set_postfix({'loss': f'{avg_loss:.4e}'})
    #         logger.info(f"Epoch {epoch+1}/{cfg.training.num_epochs} - Loss: {avg_loss:.4e}")
    rank = 0
    # Save the trained model and vocabulary (only on main process)
    if rank == 0:
        model_full_path  = full_save_path + ".pth"
        config_full_path = full_save_path + ".json"

        logger.info(f"Saving model to {full_save_path}")
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
        torch.save(pytorch_model.state_dict(), model_full_path)

        # Save the full model state
        config = {
            'vocab_config': {
                # 'serializer_config': {
                #     'num_exponent_digits': vocab.serializer.num_exponent_digits,
                #     'num_mantissa_digits': vocab.serializer.num_mantissa_digits
                # },
                'vocabulary': tokenizer.vocabulary
            },
            'model_config': {
                'encoder_config': {
                    'input_dim':    cfg.model.encoder.input_dim,
                    'output_dim':   cfg.model.encoder.output_dim,
                    'support_size': cfg.model.encoder.support_size,
                    'emb_dim':      cfg.model.encoder.emb_dim
                },
                'decoder_config': {
                    'units': cfg.model.decoder.units,
                    'num_transformer_layers': cfg.model.decoder.num_transformer_layers,
                    'num_attention_heads':    cfg.model.decoder.num_attention_heads,
                    'dropout': cfg.model.decoder.dropout,
                    'emb_dim': cfg.model.decoder.emb_dim,
                    'seq_dim': cfg.model.decoder.seq_dim
                },
            },
            'training_config': {
                'data_dir':     cfg.data.data_dir,
                'num_epochs':   cfg.training.num_epochs,
                'batch_size':   cfg.training.batch_size,
                'lr':           cfg.training.learning_rate,
                'feature_dim':  cfg.data.dataset.feature_dim,
                'world_size':   cfg.training.world_size,
                'use_amp':      cfg.training.use_amp,
                'grad_accumulation': cfg.training.grad_accumulation
            }
        }
        with open(config_full_path, "w") as f:
            json.dump(config, f, indent=2)
        # TODO: all params included in decoder + encoder 
        decoder_total_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
        encoder_total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {decoder_total_params + encoder_total_params}")
        logger.info(f"Total size on disk (MB): {os.path.getsize(model_full_path) / (1024 * 1024):.2f}")
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
    python train_regression_singletarget_hydra.py

    # Use fast training configuration
    python train_regression_singletarget_hydra.py training=fast

    # Use fast training and fast model configuration
    python train_regression_singletarget_hydra.py training=fast decoder=fast encoder=fast

    # Override specific parameters
    python train_regression_singletarget_hydra.py training.num_epochs=50 training.batch_size=8

    # Use different data directory
    python train_regression_singletarget_hydra.py data_dir=./path/to/data
    """

    logger.info("Using GPU training mode")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")

    # Set world_size if not specified
    if cfg.training.world_size is None:
        cfg.training.world_size = torch.cuda.device_count()

    #TODO: logger.info(f"Starting distributed training with {cfg.training.world_size} GPUs")
    # For GPU training, run directly without multiprocessing
    training_routine(cfg)

if __name__ == "__main__":
    main()