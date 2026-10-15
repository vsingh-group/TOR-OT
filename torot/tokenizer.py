"""Serialisation of sparse transport plans into token sequences.

Sequence format (one ``ROW`` block per source bin, in source-index order; entries in
target-index order):

    BOS  ( ROW  ( <tgt j>  <E-k>  d1 .. dD )* )^n  EOS

* The source index is implicit: it is the number of ``ROW`` tokens emitted so far. A plan
  always has exactly ``n`` row blocks; a row with no entries is ``ROW`` followed directly
  by the next ``ROW`` (or by ``EOS`` for the last row).
* Row decoding (``model.decoding: row``) serialises a single row as the same format without
  the ``ROW`` token: ``BOS ( <tgt j> <E-k> d1 .. dD )* EOS`` (``encode_row`` / ``decode_row``).
* ``<tgt j>`` is a *pointer* id that refers to the j-th target-bin encoder state. Pointer
  ids are laid out after the fixed vocabulary: ``tgt j -> V + j``.
* A mass v in (0, 1] is written exponent-first in scientific notation,
  v ~= 0.d1d2..dD x 10^-(k-1), i.e. ``<E-k>`` followed by D mantissa digits with d1 != 0.
  Masses below 10^-max_exponent are dropped from the serialisation.

The grammar is deterministic, so the *kind* of every slot (structural, pointer, exponent,
p-th digit) is known in advance. Kinds drive the significance-weighted loss and the
constrained decoding masks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

# slot kinds (targets); digit p (1-based) has kind KIND_DIGIT + p - 1
KIND_PAD = -1
KIND_STRUCT = 0
KIND_TGT = 1
KIND_EXP = 2
KIND_DIGIT = 3

# grammar phases: what the *next* token must be
PH_START = 0        # after BOS: expect ROW
PH_ROW = 1          # after ROW: expect a target pointer, the next ROW, or EOS
PH_TGT = 2          # after a target pointer: expect an exponent
PH_DIGITS = 3       # after exponent / digit: expect a digit
PH_ENTRY_DONE = 4   # after the last digit: expect a target pointer, ROW or EOS
PH_DONE = 5         # after EOS: expect PAD


@dataclass
class GrammarState:
    phase: int = PH_START
    row: int = -1           # index of the row currently being emitted (= #ROW tokens - 1)
    digits_seen: int = 0
    last_tgt: int = -1      # last target index emitted in the current row

    @property
    def finished(self) -> bool:
        return self.phase == PH_DONE


class PlanTokenizer:
    PAD, BOS, EOS, ROW = 0, 1, 2, 3
    NUM_SPECIAL = 4

    def __init__(self, mantissa_digits: int = 4, max_exponent: int = 9):
        if mantissa_digits < 1 or max_exponent < 1:
            raise ValueError("mantissa_digits and max_exponent must be >= 1")
        self.D = mantissa_digits
        self.max_exponent = max_exponent
        self.digit0 = self.NUM_SPECIAL                   # ids of digits 0..9
        self.exp0 = self.digit0 + 10                     # ids of <E-1> .. <E-max_exponent>
        self.vocab_size = self.exp0 + max_exponent       # V: fixed vocabulary (pointers follow)

    # ------------------------------------------------------------------ id helpers
    def digit_id(self, d: int) -> int:
        return self.digit0 + d

    def exponent_id(self, k: int) -> int:
        """Id of <E-k>, k in [1, max_exponent]."""
        return self.exp0 + (k - 1)

    def is_digit(self, t: int) -> bool:
        return self.digit0 <= t < self.digit0 + 10

    def is_exponent(self, t: int) -> bool:
        return self.exp0 <= t < self.exp0 + self.max_exponent

    def tgt_id(self, j: int) -> int:
        return self.vocab_size + j

    def is_pointer(self, t: int) -> bool:
        return t >= self.vocab_size

    def extended_size(self, m_pad: int) -> int:
        return self.vocab_size + m_pad

    # ------------------------------------------------------------------ mass <-> tokens
    def encode_mass(self, v: float) -> Tuple[int, str] | None:
        """(k, 'd1..dD') with v ~= 0.d1..dD x 10^-(k-1); None if v is too small to represent."""
        if not (0.0 < v <= 1.0):
            raise ValueError(f"mass must be in (0, 1], got {v}")
        k = -math.floor(math.log10(v))          # v in [10^-k, 10^-(k-1))  ->  0.d1.. x 10^-(k-1)
        mant = round(v * 10 ** (k + self.D - 1))  # integer with D digits, first digit != 0
        if mant >= 10**self.D:                   # rounding overflow, e.g. 0.099996 -> 0.1000
            k -= 1
            mant = 10 ** (self.D - 1)
        if k < 1:                                # v rounds to 1.0: clamp to the largest representable value
            k, mant = 1, 10**self.D - 1
        if k > self.max_exponent:
            return None
        return k, f"{mant:0{self.D}d}"

    def decode_mass(self, k: int, digits: Sequence[int]) -> float:
        mant = 0
        for d in digits:
            mant = mant * 10 + d
        return mant * 10.0 ** (-(k + self.D - 1))

    # ------------------------------------------------------------------ plan <-> ids
    def encode(
        self,
        rows: Sequence[int],
        cols: Sequence[int],
        vals: Sequence[float],
        n: int,
    ) -> Tuple[List[int], List[int]]:
        """Serialise a COO plan with ``n`` source bins. Returns (ids, kinds), both starting
        with BOS and ending with EOS."""
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        vals = np.asarray(vals, dtype=np.float64)
        order = np.lexsort((cols, rows))

        ids: List[int] = [self.BOS]
        kinds: List[int] = [KIND_STRUCT]
        current_row = -1
        for idx in order:
            enc = self.encode_mass(float(vals[idx]))
            if enc is None:
                continue
            k, digits = enc
            while current_row < rows[idx]:          # open this row (and any empty rows before it)
                ids.append(self.ROW)
                kinds.append(KIND_STRUCT)
                current_row += 1
            ids.append(self.tgt_id(int(cols[idx])))
            kinds.append(KIND_TGT)
            ids.append(self.exponent_id(k))
            kinds.append(KIND_EXP)
            for p, ch in enumerate(digits):
                ids.append(self.digit_id(int(ch)))
                kinds.append(KIND_DIGIT + p)
        while current_row < n - 1:                  # trailing empty rows
            ids.append(self.ROW)
            kinds.append(KIND_STRUCT)
            current_row += 1
        ids.append(self.EOS)
        kinds.append(KIND_STRUCT)
        return ids, kinds

    def decode(self, ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse ids back into a COO plan (rows, cols, vals). Tolerant: stops at the first
        token that is not admissible in the current phase, or at EOS."""
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        phase = PH_START
        row = -1
        col = -1
        k = 0
        digits: List[int] = []
        for t in ids:
            t = int(t)
            if phase == PH_START and t == self.BOS:
                continue
            if phase in (PH_START, PH_ROW, PH_ENTRY_DONE) and t == self.ROW:
                row += 1
                phase = PH_ROW
            elif phase in (PH_ROW, PH_ENTRY_DONE) and self.is_pointer(t):
                col = t - self.vocab_size
                phase = PH_TGT
            elif phase == PH_TGT and self.is_exponent(t):
                k = t - self.exp0 + 1
                digits = []
                phase = PH_DIGITS
            elif phase == PH_DIGITS and self.is_digit(t):
                digits.append(t - self.digit0)
                if len(digits) == self.D:
                    rows.append(row)
                    cols.append(col)
                    vals.append(self.decode_mass(k, digits))
                    phase = PH_ENTRY_DONE
            elif phase in (PH_ROW, PH_ENTRY_DONE) and t == self.EOS:
                break
            else:
                break
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64), np.asarray(vals, dtype=np.float64)

    def encode_row(self, cols: Sequence[int], vals: Sequence[float]) -> Tuple[List[int], List[int]]:
        """Serialise a single row: ``BOS ( <tgt j> <E-k> d1 .. dD )* EOS``."""
        ids, kinds = self.encode(np.zeros(len(cols), dtype=np.int64), cols, vals, n=1)
        return ids[:1] + ids[2:], kinds[:1] + kinds[2:]          # drop the ROW token

    def decode_row(self, ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Parse a single-row sequence (with or without the leading BOS) into (cols, vals)."""
        ids = [int(t) for t in ids]
        if ids and ids[0] == self.BOS:
            ids = ids[1:]
        _, cols, vals = self.decode([self.ROW] + ids)
        return cols, vals

    # ------------------------------------------------------------------ grammar (constrained decoding)
    @staticmethod
    def row_state() -> GrammarState:
        """Initial grammar state for a single-row sequence; use ``allowed_mask`` with ``n=1``."""
        return GrammarState(phase=PH_ROW, row=0)

    def advance(self, state: GrammarState, t: int) -> None:
        """Update ``state`` in place after emitting token ``t`` (assumed admissible)."""
        if t == self.ROW and state.phase in (PH_START, PH_ROW, PH_ENTRY_DONE):
            state.row += 1
            state.last_tgt = -1
            state.phase = PH_ROW
        elif t == self.EOS and state.phase in (PH_ROW, PH_ENTRY_DONE):
            state.phase = PH_DONE
        elif self.is_pointer(t) and state.phase in (PH_ROW, PH_ENTRY_DONE):
            state.last_tgt = t - self.vocab_size
            state.phase = PH_TGT
        elif state.phase == PH_TGT:
            state.digits_seen = 0
            state.phase = PH_DIGITS
        elif state.phase == PH_DIGITS:
            state.digits_seen += 1
            if state.digits_seen == self.D:
                state.phase = PH_ENTRY_DONE
        elif state.phase == PH_DONE:
            pass
        else:
            raise ValueError(f"token {t} not admissible in phase {state.phase}")

    def allowed_mask(self, state: GrammarState, n: int, m: int, m_pad: int) -> np.ndarray:
        """Boolean mask over the extended vocabulary of admissible next tokens for an
        instance with ``n`` source and ``m`` target bins.

        Besides the grammar it enforces the serialisation convention: exactly ``n`` ROW
        blocks, and strictly increasing target indices within a row.
        """
        mask = np.zeros(self.extended_size(m_pad), dtype=bool)
        tgt0 = self.vocab_size
        if state.phase == PH_START:
            mask[self.ROW] = True
        elif state.phase in (PH_ROW, PH_ENTRY_DONE):
            mask[tgt0 + state.last_tgt + 1 : tgt0 + m] = True
            if state.row < n - 1:
                mask[self.ROW] = True
            else:
                mask[self.EOS] = True
        elif state.phase == PH_TGT:
            mask[self.exp0 : self.exp0 + self.max_exponent] = True
        elif state.phase == PH_DIGITS:
            mask[self.digit0 : self.digit0 + 10] = True
            if state.digits_seen == 0:
                mask[self.digit0] = False           # leading mantissa digit is never 0
        else:
            mask[self.PAD] = True
        return mask
