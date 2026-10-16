"""PyTorch access to the Hugging Face corpus: instances, transpose augmentation, batching.

Two views of the corpus:

* ``OTPlanDataset`` (``model.decoding: plan``): one example per instance, target = whole plan.
* ``OTRowDataset`` (``model.decoding: row``): one example per (instance, row i). Rows
  0..i-1 are taken as fixed, and the example is the *remaining* problem: source bins
  i..n-1 and the remaining column capacities r = b - sum of rows 0..i-1 as target masses.
  Row i of the optimal plan is the first row of an optimal plan of that remaining problem,
  which is the target. The last row is never predicted (it equals the final capacities).
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Iterator, List, Sequence

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


def dense_plan(inst: Instance) -> np.ndarray:
    P = np.zeros((inst["n"], inst["m"]), dtype=np.float64)
    np.add.at(P, (inst["plan_rows"], inst["plan_cols"]), inst["plan_vals"])
    return P


def row_subproblem(inst: Instance, i: int, capacity: np.ndarray) -> Instance:
    """The remaining problem after rows 0..i-1: source bins i..n-1, target masses ``capacity``.

    Target bins keep their original indices (exhausted ones get mass 0), so pointer ids in
    the row sequence refer to the original columns. The returned dict has no plan.
    """
    return {
        "n": inst["n"] - i,
        "m": inst["m"],
        "dim": inst["dim"],
        "cost_type": inst["cost_type"],
        "coords_src": inst["coords_src"][i:],
        "coords_tgt": inst["coords_tgt"],
        "a": inst["a"][i:],
        "b": np.asarray(capacity, dtype=np.float64),
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

    tokens_per_size = 6.5   # decoder tokens per unit of ``sizes()`` (a plan has ~6.5 (n + m) tokens)

    def sizes(self) -> np.ndarray:
        """n + m per instance (for bucketing), read column-wise without materialising records."""
        return np.asarray(self.split["n"], dtype=np.int64) + np.asarray(self.split["m"], dtype=np.int64)


class OTRowDataset(Dataset):
    """One example per (instance, row i), i = 0..n-2: the remaining problem and row i as target.

    Two optional perturbations make training capacities look like the slightly wrong ones the
    model produces itself at inference time (rows are subtracted after being predicted):

    * ``residual_leftover_prob``: each used-up column (exact capacity 0) gets, with this
      probability, a small leftover drawn log-uniformly from ``residual_leftover_range``. The
      target row is unchanged, so the model learns to ignore such leftovers instead of
      routing mass into them.
    * ``residual_noise``: the other capacities are multiplied by ``1 + noise * N(0, 1)``.

    Transposition is drawn per example, like in ``OTPlanDataset``.
    """

    tokens_per_size = 1.0   # ``sizes()`` already counts encoder tokens

    def __init__(
        self,
        hf_split,
        transpose_aug_prob: float = 0.0,
        residual_noise: float = 0.0,
        residual_leftover_prob: float = 0.0,
        residual_leftover_range: Sequence[float] = (1e-6, 1e-2),
    ):
        self.split = hf_split
        self.transpose_aug_prob = transpose_aug_prob
        self.residual_noise = residual_noise
        self.residual_leftover_prob = residual_leftover_prob
        self.log_leftover_range = np.log10(np.asarray(residual_leftover_range, dtype=np.float64))
        self.cost_map = cost_label_map(hf_split)
        self.n = np.asarray(hf_split["n"], dtype=np.int64)
        self.m = np.asarray(hf_split["m"], dtype=np.int64)
        rows = np.maximum(self.n - 1, 0)
        self.starts = np.concatenate([[0], np.cumsum(rows)[:-1]])
        self.rows = rows

    def __len__(self) -> int:
        return int(self.rows.sum())

    def locate(self, idx: int) -> tuple[int, int]:
        """(instance index, row index) of example ``idx``."""
        k = int(np.searchsorted(self.starts, idx, side="right") - 1)
        return k, int(idx - self.starts[k])

    def __getitem__(self, idx: int) -> Instance:
        k, i = self.locate(int(idx))
        inst = record_to_instance(self.split[k], self.cost_map)
        if self.transpose_aug_prob > 0.0 and random.random() < self.transpose_aug_prob:
            inst = transpose_instance(inst)
            i = i % max(1, inst["n"] - 1)
        P = dense_plan(inst)
        capacity = np.maximum(inst["b"] - P[:i].sum(axis=0), 0.0)
        used_up = capacity <= 1e-12
        if self.residual_noise > 0.0:
            noise = 1.0 + self.residual_noise * np.random.standard_normal(capacity.shape)
            capacity = np.maximum(capacity * noise, 0.0)
        if self.residual_leftover_prob > 0.0:
            hit = used_up & (np.random.random(capacity.shape) < self.residual_leftover_prob)
            leftover = 10.0 ** np.random.uniform(*self.log_leftover_range, size=capacity.shape)
            capacity = np.where(hit, leftover, capacity)
        sub = row_subproblem(inst, i, capacity)
        cols = np.nonzero(P[i] > 0.0)[0]
        sub["row_cols"], sub["row_vals"] = cols, P[i, cols]
        return sub

    def sizes(self) -> np.ndarray:
        """Encoder tokens of every example, (n - i) + m, without materialising records."""
        i = np.arange(len(self)) - np.repeat(self.starts, self.rows)
        return np.repeat(self.n + self.m, self.rows) - i


class Collator:
    """Pads a list of instances to a batch and tokenises the targets.

    The target is the whole plan (``plan_*`` keys), a single row (``row_cols``/``row_vals``,
    row decoding), or absent (inference: the sequence is just ``BOS``).
    """

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
            if "row_cols" in inst:
                ids, kinds = self.tok.encode_row(inst["row_cols"], inst["row_vals"])
            elif "plan_rows" in inst:
                ids, kinds = self.tok.encode(inst["plan_rows"], inst["plan_cols"], inst["plan_vals"], ni)
            else:
                ids, kinds = [self.tok.BOS], [KIND_PAD]
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
        counts = np.arange(1, self.batch_size + 1)
        batches, start = [], 0
        while start < len(order):
            # running max: order is sorted, but keys are noisy; count * longest grows with the
            # prefix length, so the batch is the number of prefixes within the token cap
            longest = np.maximum.accumulate(est[start : start + self.batch_size])
            fits = int(np.count_nonzero(counts[: len(longest)] * longest <= self.max_tokens))
            end = start + max(fits, 1)
            batches.append(order[start:end])
            start = end
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


DECODER_KEYS = ("input_ids", "target_ids", "kinds")


def split_batch(batch: Dict[str, Any], max_tokens: int | None) -> List[Dict[str, Any]]:
    """Cut a collated batch into micro-batches of at most ``max_tokens`` padded decoder tokens.

    A batch is padded to its longest sequence, so a few long rows can make it far larger than
    the sampler's estimate (in row mode the sampler only counts encoder tokens). Sequences are
    grouped by length and each micro-batch is trimmed to its own longest one.
    """
    B, T = batch["input_ids"].shape
    if max_tokens is None or B * T <= max_tokens:
        return [batch]
    lengths = (batch["kinds"] != KIND_PAD).sum(dim=1)
    order = torch.argsort(lengths, descending=True)
    micro, start = [], 0
    while start < B:
        longest = max(int(lengths[order[start]]), 1)
        idx = order[start : start + max(1, max_tokens // longest)]
        mb = {k: v[idx] for k, v in batch.items()}
        for k in DECODER_KEYS:
            mb[k] = mb[k][:, :longest]
        micro.append(mb)
        start += len(idx)
    return micro
