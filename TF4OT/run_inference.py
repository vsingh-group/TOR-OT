from TF4OT import tokenizers
from TF4OT import model
from TF4OT import utils

#from TF4OT.serialization.numeric import tokens
#from TF4OT.multitarget_models import *
#from TF4OT.datasets import load_random_uniform_data_1d, load_ot_data_single_target_format
#rom TF4OT.utils.multitarget_datasets_hf import *
from datasets import load_dataset
import torch
import functools
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
import json
import os

# Add distributed training imports
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Add Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# Setup logging
logger = logging.getLogger(__name__)

def inference_routine(cfg):
    """
    Do inference on the saved model.
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        force=True)

    device = torch.device(
        getattr(cfg.inference, "device", "cuda" if torch.cuda.is_available() else "cpu")
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
    logger.info(f"Model will be loaded from: {full_save_path}")

    pytorch_model, tokenizer = load_model_by_name(full_save_path)

    # Loading TEST dataset
    logger.info("Loading dataset: " + cfg.data.hf_dataset)
    dataset     = load_dataset(cfg.data.hf_dataset, cache_dir=cfg.data.hf_cache_dir)
    test_data   = dataset["test"]
    test_loader = DataLoader(test_data,
                            batch_size=cfg.training.batch_size,
                            shuffle=False,
                            num_workers=cfg.data.loading.num_workers,
                            pin_memory=(device.type == "cuda"),
                            collate_fn=utils.collate_from_string
    )
    logger.info(f"test_size={len(test_data)}, test.batch_size={cfg.training.batch_size}, "
                f"batches_per_epoch={len(test_loader)}")

    # Create Tokenizer
    logger.info("Creating Tokenizer")
    # TODO: add support for other tokenizers
    OT_INPUT_TOTAL         = "input_total"
    OT_INPUT_COST_MATRIX   = "input_cost_matrix"
    OT_TARGET_PLAN_MATRIX  = "target_otplan_solve_sparse"
    OT_TARGET_PLAN_MATRIX2 = "output_map_sparse"

    pytorch_model.eval()
    with torch.no_grad():
        pbar = tqdm(test_loader, total=len(test_loader))
        for batch in pbar:

            encoder_input_ab = batch[OT_INPUT_TOTAL]
            encoder_input_ab = encoder_input_ab.to(device)

            text_input, pad_mask = tokenizer.to_tokens(batch[OT_TARGET_PLAN_MATRIX], drop_last=False)

            pad_mask      = torch.cat([torch.full((encoder_input_ab.shape[0], 1), True), pad_mask], dim=1)
            text_ids_pred = pytorch_model.generate(encoder_input_ab,
                                                    temperature=0.1,
                                                    max_seq_len=cfg.model.decoder.seq_dim)
            text_pred = tokenizer.from_tokens(text_ids_pred)

            for target, pred in zip(utils.to_string_lists(batch[OT_TARGET_PLAN_MATRIX], cfg.tokenizer.decimal_precision), text_pred):
                print ("target: ", target)
                print ("pred:   ", pred[:pred.find(']')+1])
                print ('====================================')
            break
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

    # # Cleanup distributed training
    # cleanup_distributed(use_cpu)
    # return (decoder.module if not use_cpu else decoder), vocab

def load_model_by_name(full_save_path: str, device: str = "cuda"):
    """
    Load a TokenRegressor saved in:
      artifacts/{name}/model_weights.pth
      artifacts/{name}/config.json

    Returns: (model, tokenizer)
    """
    model_full_path  = full_save_path + ".pth"
    config_full_path = full_save_path + ".json"

    with open(config_full_path) as f:
        cfg = json.load(f)

    # Extract dimensions from new automatic config structure
    input_total_dim       = cfg["data"]["dataset"]["input_total_dim"]
    input_cost_matrix_dim = cfg["data"]["dataset"]["input_cost_matrix_dim"]

    # Rebuild tokenizer from config
    tokenizer_cfg = cfg["tokenizer"]#["IEEEFloatTokenizer"]
    
    # TODO: add support for automatic tokenizer selection
    # For now, assume IEEEFloatTokenizer based on config keys
    if "base" in tokenizer_cfg:
        tokenizer = tokenizers.IEEEFloatTokenizer(
            base=tokenizer_cfg["base"],
            num_exponent_digits=tokenizer_cfg["num_exponent_digits"],
            num_mantissa_digits=tokenizer_cfg["num_mantissa_digits"]
        )
    elif "vocabulary" in tokenizer_cfg:
        tokenizer = tokenizers.MyDigitTokenizer(
            vocabulary=tokenizer_cfg["vocabulary"],
            decimal_precision=tokenizer_cfg["decimal_precision"]
        )
    else:
        raise ValueError("Unknown tokenizer configuration")

    # Rebuild encoder
    emb_dim = cfg["model"]["encoder"]["emb_dim"]
    encoder_ab = nn.Sequential(
        nn.Linear(input_total_dim, emb_dim),
        nn.GELU(),
        nn.Linear(emb_dim, emb_dim)
    ).to(device)
    
    encoder_C = nn.Sequential(
        nn.Linear(input_cost_matrix_dim, emb_dim),
        nn.GELU(),
        nn.Linear(emb_dim, emb_dim)
    ).to(device)

    # Rebuild model using decoder config
    decoder_cfg = cfg["model"]["decoder"]
    pytorch_model = model.TokenRegressor(
        tokenizer   = tokenizer,
        encoder_ab  = encoder_ab,
        encoder_C   = encoder_C,
        emb_dim     = decoder_cfg["emb_dim"],
        seq_dim     = decoder_cfg["seq_dim"],
        num_transformer_layers = decoder_cfg["num_transformer_layers"],
        num_attention_heads    = decoder_cfg["num_attention_heads"],
        dropout     = decoder_cfg.get("dropout", 0.1)  # Use default if not present
    ).to(device)

    # Load weights
    model_state = torch.load(model_full_path, map_location=device, weights_only=True)
    pytorch_model.load_state_dict(model_state, strict=True)

    return pytorch_model, tokenizer

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main function using Hydra for configuration management.
    """

    logger.info("Using GPU inference mode")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")

    # Set world_size if not specified
    if cfg.training.world_size is None: cfg.training.world_size = torch.cuda.device_count()

    #TODO: logger.info(f"Starting distributed training with {cfg.training.world_size} GPUs")
    # For GPU training, run directly without multiprocessing
    inference_routine(cfg)

if __name__ == "__main__":
    main()