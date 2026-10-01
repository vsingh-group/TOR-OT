"""PyTorch access to the Hugging Face corpus: instances, transpose augmentation, batching."""

from __future__ import annotations

import random
from typing import Any, Dict, Iterator, List, Sequence

import os

import numpy as np
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import Dataset, Sampler

from torot.data.cost import COST_TYPES
from torot.data.supports import pad_coords
from torot.tokenizer import KIND_PAD, PlanTokenizer

Instance = Dict[str, Any]


def load_corpus(repo_id: str, cache_dir: str | None = None) -> DatasetDict:
    """Load the corpus from the Hub, or from a local ``save_to_disk`` directory."""
    if os.path.isdir(repo_id):
        return load_from_disk(repo_id)
    return load_dataset(repo_id, cache_dir=cache_dir)


def cost_label_map(hf_split) -> List[int]:
    """Map the split's ``cost_type`` class labels onto ``COST_TYPES`` indices.

    Raises if the corpus was generated with a different cost set, so that a stale corpus
    can never be silently trained on or evaluated with the wrong cost matrices.
    """
    names = list(hf_split.features["cost_type"].names)
    unknown = [n for n in names if n not in COST_TYPES]
    if unknown:
        raise ValueError(
            f"corpus cost types {names} do not match torot.data.cost.COST_TYPES {list(COST_TYPES)} "
            f"(unknown: {unknown}); regenerate the corpus with torot.data.generate"
        )
    return [COST_TYPES.index(n) for n in names]


def record_to_instance(rec: dict, cost_map: Sequence[int] | None = None) -> Instance:
    """Convert a raw HF record (lists) into numpy arrays."""
    cost = int(rec["cost_type"])
    return {
        "n": int(rec["n"]),
        "m": int(rec["m"]),
        "dim": int(rec["dim"]),
        "cost_type": cost if cost_map is None else int(cost_map[cost]),
        "coords_src": pad_coords(np.asarray(rec["coords_src"], dtype=np.float64)),
        "coords_tgt": pad_coords(np.asarray(rec["coords_tgt"], dtype=np.float64)),
        "a": np.asarray(rec["a"], dtype=np.float64),
        "b": np.asarray(rec["b"], dtype=np.float64),
        "plan_rows": np.asarray(rec["plan_rows"], dtype=np.int64),
        "plan_cols": np.asarray(rec["plan_cols"], dtype=np.int64),
        "plan_vals": np.asarray(rec["plan_vals"], dtype=np.float64),
        "ot_cost": float(rec["ot_cost"]),
    }


def transpose_instance(inst: Instance) -> Instance:
    """Symmetry 6.1: pi*(b, a) = pi*(a, b)^T for symmetric costs. Swaps source and target."""
    return {
        **inst,
        "n": inst["m"],
        "m": inst["n"],
        "coords_src": inst["coords_tgt"],
        "coords_tgt": inst["coords_src"],
        "a": inst["b"],
        "b": inst["a"],
        "plan_rows": inst["plan_cols"],
        "plan_cols": inst["plan_rows"],
    }


class OTPlanDataset(Dataset):
    """Wraps one split of the HF dataset; optionally applies the transpose augmentation.

    The augmentation draw uses Python's ``random`` module, which ``DataLoader`` seeds
    independently in every worker process.
    """

    def __init__(self, hf_split, transpose_aug_prob: float = 0.0):
        self.split = hf_split
        self.transpose_aug_prob = transpose_aug_prob
        self.cost_map = cost_label_map(hf_split)

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> Instance:
        inst = record_to_instance(self.split[int(idx)], self.cost_map)
        if self.transpose_aug_prob > 0.0 and random.random() < self.transpose_aug_prob:
            inst = transpose_instance(inst)
        return inst

    def sizes(self) -> np.ndarray:
        """n + m per instance (for bucketing), read column-wise without materialising records."""
        return np.asarray(self.split["n"], dtype=np.int64) + np.asarray(self.split["m"], dtype=np.int64)


class Collator:
    """Pads a list of instances to a batch and tokenises the plans."""

    def __init__(self, tokenizer: PlanTokenizer):
        self.tok = tokenizer

    def __call__(self, instances: Sequence[Instance]) -> Dict[str, Any]:
        B = len(instances)
        n_pad = max(inst["n"] for inst in instances)
        m_pad = max(inst["m"] for inst in instances)

        a = torch.zeros(B, n_pad, dtype=torch.float32)
        b = torch.zeros(B, m_pad, dtype=torch.float32)
        coords_src = torch.zeros(B, n_pad, 2, dtype=torch.float32)
        coords_tgt = torch.zeros(B, m_pad, 2, dtype=torch.float32)
        src_mask = torch.zeros(B, n_pad, dtype=torch.bool)
        tgt_mask = torch.zeros(B, m_pad, dtype=torch.bool)
        n = torch.zeros(B, dtype=torch.long)
        m = torch.zeros(B, dtype=torch.long)
        cost_type = torch.zeros(B, dtype=torch.long)

        sequences: List[List[int]] = []
        kinds_list: List[List[int]] = []
        for i, inst in enumerate(instances):
            ni, mi = inst["n"], inst["m"]
            a[i, :ni] = torch.from_numpy(inst["a"].astype(np.float32))
            b[i, :mi] = torch.from_numpy(inst["b"].astype(np.float32))
            coords_src[i, :ni] = torch.from_numpy(inst["coords_src"]).float()
            coords_tgt[i, :mi] = torch.from_numpy(inst["coords_tgt"]).float()
            src_mask[i, :ni] = True
            tgt_mask[i, :mi] = True
            n[i], m[i], cost_type[i] = ni, mi, inst["cost_type"]
            ids, kinds = self.tok.encode(inst["plan_rows"], inst["plan_cols"], inst["plan_vals"], ni)
            sequences.append(ids)
            kinds_list.append(kinds)

        T = max(len(s) for s in sequences)
        ids = torch.full((B, T), self.tok.PAD, dtype=torch.long)
        kinds = torch.full((B, T), KIND_PAD, dtype=torch.long)
        for i, (s, k) in enumerate(zip(sequences, kinds_list)):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            kinds[i, : len(k)] = torch.tensor(k, dtype=torch.long)

        return {
            "a": a,
            "b": b,
            "coords_src": coords_src,
            "coords_tgt": coords_tgt,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
            "n": n,
            "m": m,
            "cost_type": cost_type,
            "input_ids": ids[:, :-1],
            "target_ids": ids[:, 1:],
            "kinds": kinds[:, 1:],
            "instances": list(instances),
        }


class SizeBucketBatchSampler(Sampler[List[int]]):
    """Batches instances of similar size (n + m) to limit padding; DDP-aware.

    A batch holds at most ``batch_size`` instances and at most ``max_tokens`` *padded*
    decoder tokens (instances x longest sequence in the batch), estimated from the support
    sizes: a plan with n + m bins serialises to about ``tokens_per_bin * (n + m)`` tokens.
    """

    def __init__(
        self,
        sizes: np.ndarray,
        batch_size: int,
        max_tokens: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        bucket_noise: float = 0.1,
        tokens_per_bin: float = 6.5,
    ):
        self.sizes = np.asarray(sizes)
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.bucket_noise = bucket_noise
        self.tokens_per_bin = tokens_per_bin
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _split(self, order: np.ndarray) -> List[np.ndarray]:
        """Greedily cut the size-sorted ``order`` into batches under both caps."""
        if self.max_tokens is None:
            return [order[i : i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        est = self.tokens_per_bin * self.sizes[order] + 2
        batches, start = [], 0
        for i in range(1, len(order) + 1):
            count = i - start
            longest = est[start:i].max()          # order is sorted, but keys are noisy
            if count > self.batch_size or count * longest > self.max_tokens:
                batches.append(order[start : i - 1])
                start = i - 1
        if start < len(order):
            batches.append(order[start:])
        return batches

    def _batches(self) -> List[np.ndarray]:
        rng = np.random.default_rng(self.seed + self.epoch)
        keys = self.sizes.astype(np.float64)
        if self.shuffle:
            keys = keys * (1.0 + rng.uniform(-self.bucket_noise, self.bucket_noise, size=keys.shape))
        order = np.argsort(keys, kind="stable")
        batches = self._split(order)
        if self.shuffle:
            rng.shuffle(batches)
        # every rank sees the same number of batches
        usable = (len(batches) // self.world_size) * self.world_size
        return batches[:usable][self.rank :: self.world_size]

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self._batches():
            yield batch.tolist()

    def __len__(self) -> int:
        return len(self._batches())


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
