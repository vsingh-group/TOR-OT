"""Custom tokenizers for TF4OT.

For full generality, note that the tokens are strings, as we assume the
vocabulary will assign integer IDs.
"""

from typing import Generic, TypeVar
from TF4OT import utils
import numpy as np
import ordered_set
import torch
import abc
import math
import re

OrderedSet = ordered_set.OrderedSet
DELIMITERS = ('<', '>')

def _to_token(s: str | int) -> str:
  left_d, right_d = DELIMITERS
  return f'{left_d}{s}{right_d}'

def _from_token(token: str) -> str | int:
  left_d, right_d = DELIMITERS
  pattern = f'{left_d}(.*?){right_d}'
  m = re.fullmatch(pattern, token)
  if not m:
    raise ValueError(f'Could not deserialize `{token}`.')
  return m.group(1)


ObjectT = TypeVar('ObjectT')


class DecoderTokenizer(abc.ABC, Generic[ObjectT]):
  """Abstract class for decoder tokenizers."""

  @property
  @abc.abstractmethod
  def num_tokens_per_obj(self) -> int:
    """Number of tokens used to represent each float."""
  
  @property
  @abc.abstractmethod
  def vocab_size(self) -> int:
    """Size of the vocabulary."""

  @abc.abstractmethod
  def all_tokens(self) -> OrderedSet[str]:
    """Returns ordered set of all tokens used."""

  @abc.abstractmethod
  def possible_next_tokens(self, prev_tokens: list[str]) -> OrderedSet[str]:
    """Returns ordered set of tokens possible for the next step."""

  @abc.abstractmethod
  def to_tokens(self, obj: ObjectT, /) -> list[str]:
    """Converts an object to a string of tokens."""

  @abc.abstractmethod
  def from_tokens(self, tokens: list[str], /) -> ObjectT:
    """Converts a string of tokens to an object."""


# =============================================================================
# DigitTokenizer: fixed-length character tokenizer (IEEE-style structure)
# =============================================================================
class DigitTokenizer(DecoderTokenizer[str]):
  """Fixed-length character tokenizer over a given alphabet.

  Encodes a string into exactly `length` tokens drawn from `alphabet`,
  padding with a dedicated `pad_symbol` when the string is shorter.

  Notes:
    - Tokens are serialized with _to_token(...) (e.g., '<0>', '<PAD>').
    - All positions allow any alphabet symbol (like NormalizedTokenizer).
    - `from_tokens` strips trailing pad symbols when reconstructing the string.

  Args:
    alphabet: Iterable of symbols (e.g., ['0', '1', ..., '9']) or a string
      of symbols (e.g., '0123456789.-').
    length: Number of tokens per object (fixed).
    pad_symbol: Symbol used internally to pad up to `length`. It is part of the
      tokenizer's token set so it can appear in `from_tokens`.

  Example:
    >>> tok = DigitTokenizer(alphabet='0123456789', length=5, pad_symbol='PAD')
    >>> tok.to_tokens('123')
    ['<1>', '<2>', '<3>', '<PAD>', '<PAD>']
    >>> tok.from_tokens(['<0>','<0>','<7>','<5>','<PAD>'])
    '0075'
  """

  def __init__(self,
              alphabet: list[str] | str = '0123456789',
              length: int = 8,
              pad_symbol: str = 'PAD',
              ):
    if isinstance(alphabet, str):
      alphabet = list(alphabet)
    if len(alphabet) == 0:
      raise ValueError('alphabet must be non-empty')
    if length <= 0:
      raise ValueError('length must be >= 1')

    self.alphabet: list[str] = [str(a) for a in alphabet]
    self.length: int = length
    self.pad_symbol: str = str(pad_symbol)

    # Internal ordered set of symbols including pad.
    self._symbols: OrderedSet[str] = OrderedSet(self.alphabet + [self.pad_symbol])

  @property
  def num_tokens_per_obj(self) -> int:
    return self.length
  
  @property
  def vocab_size(self) -> int:
    return len(self.all_tokens())

  def all_tokens(self) -> OrderedSet[str]:
    # Return serialized tokens (e.g., '<0>', '<PAD>').
    return OrderedSet([_to_token(s) for s in self._symbols])

  def possible_next_tokens(self, prev_tokens: list[str]) -> OrderedSet[str]:
    # Any symbol may appear at any position (similar to NormalizedTokenizer).
    if len(prev_tokens) >= self.num_tokens_per_obj:
      raise ValueError(f'Index {len(prev_tokens)} out of bounds.')
    return self.all_tokens()

  def _validate_chars(self, s: str) -> None:
    for ch in s:
      if ch not in self._symbols:
        raise ValueError(f'Character "{ch}" not in alphabet/pad set {list(self._symbols)}')

  def to_tokens(self, s: str, /) -> list[str]:
      if not isinstance(s, str):
          raise TypeError(f'DigitTokenizer expects a string, got {type(s)}')

      # Validate characters against alphabet (pad is not expected in input)
      self._validate_chars(s if s else '')
      if len(s) > self.length:
          raise ValueError(f'Input length {len(s)} exceeds fixed length {self.length}')

      # Build symbol list: characters of s + whole pad symbols
      symbols = list(s) + [self.pad_symbol] * (self.length - len(s))

      # Serialize each symbol once (so PAD stays '<PAD>')
      return [_to_token(sym) for sym in symbols]

  def from_tokens(self, tokens: list[str], /) -> str:
    if len(tokens) != self.length:
      raise ValueError(f'Expected {self.length} tokens, got {len(tokens)}')

    chars = [str(_from_token(t)) for t in tokens]
    # Validate reconstructed symbols
    for ch in chars:
      if ch not in self._symbols:
        raise ValueError(f'Token "{ch}" not in alphabet/pad set {list(self._symbols)}')

    # Strip trailing pads to recover the original (shorter) string
    while chars and chars[-1] == self.pad_symbol:
      chars.pop()
    return ''.join(chars)


class P10Tokenizer(DecoderTokenizer[float]):
  """Uses P10 tokenization from https://arxiv.org/abs/2112.01898.

  A float f can be represented as:

  `s * m * 10^e`

  where:
    s: Positive/Negative sign (+, -)
    m: Mantissa representing leading digits.
    e: Exponent.

  Attributes:
    num_digits: Number of digits in `m`. Each digit (even the leading) is
      between <0> and <9>.
    exponent_range: Controls number of exponent tokens, e.g. if 10, the exponent
      token range will be [<E-10>, <E10>], affecting the range of representable
      floats.
  """

  def __init__(self, num_digits: int = 4, exponent_range: int = 10):
    self.num_digits = num_digits
    self.exponent_range = exponent_range

  def all_tokens(self) -> OrderedSet[str]:
    tokens = [_to_token(s) for s in ['+', '-'] + list(range(10))]
    exps = [
        f'E{i}' for i in range(-self.exponent_range, self.exponent_range + 1)
    ]
    tokens.extend([_to_token(s) for s in exps])
    return OrderedSet(tokens)

  @property
  def num_tokens_per_obj(self) -> int:
    return 2 + self.num_digits

  def possible_next_tokens(self, prev_tokens: list[str]) -> OrderedSet[str]:
    index = len(prev_tokens)
    if index < 0 or index >= self.num_tokens_per_obj:
      raise ValueError(f'Index {index} out of bounds.')

    if index == 0:  # beginning
      tokens = [_to_token(s) for s in ['+', '-']]
    elif index == self.num_tokens_per_obj - 1:  # end
      exps = [
          f'E{i}' for i in range(-self.exponent_range, self.exponent_range + 1)
      ]
      tokens = [_to_token(s) for s in exps]
    else:  # middle (digit)
      tokens = [_to_token(s) for s in range(0, 10)]
    return OrderedSet(tokens)

  @property
  def _max_abs_val(self) -> float:
    """Largest representable positive number."""
    return float(self.num_digits * '9') * (10.0**self.exponent_range)

  @property
  def _min_abs_val(self) -> float:
    """Smallest representable positive number."""
    min_mantissa = float('1' + (self.num_digits - 1) * '0')
    return min_mantissa * (10 ** (-self.exponent_range))

  def _round_float(self, f: float) -> float:
    """Rounds float to the closest in-range value."""
    abs_f = abs(f)
    abs_f = min(abs_f, self._max_abs_val)
    if abs_f < self._min_abs_val:
      # Decides whether to move to 0.0 or `min_abs_val`.
      zero_or_min = round(abs_f / self._min_abs_val)
      abs_f = self._min_abs_val * zero_or_min
    return abs_f if f >= 0 else -abs_f

  def to_tokens(self, f: float, /) -> list[str]:
    f = self._round_float(f)
    s = np.format_float_scientific(
        f,
        precision=self.num_digits - 1,
        min_digits=self.num_digits - 1,
        sign=True,
    )
    # We expect numpy to produce scientific notation of the form `+2.123e+4`.
    # It will round for us and ensure leading digit isn't zero, unless the
    # number is zero.
    m = re.fullmatch('([+-])([0-9.]*)e(.*)', s)
    if not m:
      raise RuntimeError(f'Unexpected numpy notation: {s}')
    sign: str = m.group(1)
    digits = list(m.group(2).replace('.', ''))
    exp = int(m.group(3)) - len(digits) + 1 if f else 0

    out = [sign] + digits + [f'E{exp}']
    return [_to_token(s) for s in out]

  def from_tokens(self, tokens: list[str], /) -> float:
    primitives = [_from_token(t) for t in tokens]

    sign = -1 if primitives[0] == '-' else 1
    mantissa = int(''.join(map(str, primitives[1:-1])))
    exp = int(''.join(primitives[-1]).lstrip('E'))

    return float(sign * mantissa * 10**exp)


class IEEEFloatTokenizer(DecoderTokenizer[float]):
  """More official float tokenizer, minimizing the use of dedicated tokens.

  Follows IEEE-type standard.

  A float f = `s * b^e * m` can be represented as [s, e, m] from most to least
  important, where:
    s: Positive/Negative sign (+, -)
    b: Base
    e: Exponent (left-most is a sign, digits represented with base b)
    m: Mantissa (represented with base b)

  For example, 1.23456789e-222 can be represented as:

  <+><-><2><2><2><1><2><3><4>

  if b=10, num_exponent_digits=3, and num_mantissa_digits=4.
  """

  def __init__(
      self,
      base: int = 10,
      num_exponent_digits: int = 1,
      num_mantissa_digits: int = 4,
      pad_token: str = "<pad>"
  ):
    self.base = base
    self.num_exponent_digits = num_exponent_digits
    self.num_mantissa_digits = num_mantissa_digits
    self.pad_token = pad_token

    self.itos = [pad_token] + sorted(self.all_tokens())
    self.stoi = {token: i for i, token in enumerate(self.itos)}
    self.pad_id    = self.stoi[self.pad_token]

  def all_tokens(self) -> OrderedSet[str]:
    tokens = ['+', '-'] + list(range(self.base))
    return OrderedSet([_to_token(s) for s in tokens])

  @property
  def vocab_size(self) -> int:
    return len(self.itos)

  @property
  def num_tokens_per_obj(self) -> int:
    return 2 + self.num_exponent_digits + self.num_mantissa_digits

  def possible_next_tokens(self, prev_tokens: list[str]) -> OrderedSet[str]:
    index = len(prev_tokens)
    if index < 0 or index >= self.num_tokens_per_obj:
      raise ValueError(f'Index {index} out of bounds.')

    if index in [0, 1]:  # beginning
      tokens = [_to_token(s) for s in ['+', '-']]
    else:  # middle (digit)
      tokens = [_to_token(s) for s in range(self.base)]
    return OrderedSet(tokens)

  def to_tokens(self, f_list: list[list[float]], pad=True, drop_last=False) -> list[str]:
    all_lists = []
    for list in f_list:
      all_tokens = []
      for o in list: 
        all_tokens.extend(self.to_tokens_single_float(o))
      if drop_last:
        all_lists.append([self.stoi[t] for t in all_tokens[:-1]])
      else:
        all_lists.append([self.stoi[t] for t in all_tokens])

    # --- Padding logic ---
    mask = None
    if pad:
        max_len = max(len(toks) for toks in all_lists)

        padded_lists = []
        for toks in all_lists:
            pad_len = max_len - len(toks)
            padded = toks + [self.pad_id] * pad_len
            padded_lists.append(torch.LongTensor(padded))

        token_ids = torch.stack(padded_lists)
        mask = token_ids == self.pad_id
    else:
        # no padding → just return ragged lists
        token_ids = [torch.LongTensor(toks) for toks in all_lists]

    return token_ids, mask

  def to_tokens_single_float(self, f: float, /) -> list[str]:
    sign = '+' if f >= 0 else '-'
    abs_f = abs(f)
    exponent = math.floor(np.log(abs_f) / np.log(self.base)) if abs_f > 0 else 0

    exponent_sign = '+' if exponent >= 0 else '-'
    abs_exponent = abs(exponent)

    e = np.base_repr(abs_exponent, base=self.base)
    if len(e) > self.num_exponent_digits and exponent_sign == '+':
      raise ValueError(f'Overflow: Exponent {abs_exponent} too large.')
    if len(e) > self.num_exponent_digits and exponent_sign == '-':
      # Underflow.
      all_zeros = ['0'] * (self.num_exponent_digits + self.num_mantissa_digits)
      out = [sign, '-'] + all_zeros
      return [_to_token(s) for s in out]
    e = e.zfill(self.num_exponent_digits)

    mantissa = np.base_repr(
        int(round(abs_f * self.base ** (self.num_mantissa_digits - 1 - exponent))),
        base=self.base,
    )

    if len(mantissa) > self.num_mantissa_digits:
      mantissa = mantissa[: self.num_mantissa_digits]
    if len(mantissa) < self.num_mantissa_digits:  # Right-pad with zeros.
      mantissa += '0' * (self.num_mantissa_digits - len(mantissa))

    raw_str = sign + exponent_sign + e + mantissa
    return [_to_token(s) for s in raw_str]

  def from_tokens(self, tokens: list[str], /) -> float:
    primitives = [_from_token(t) for t in tokens]

    sign = -1 if primitives[0] == '-' else 1

    exponent_sign = -1 if primitives[1] == '-' else 1

    abs_exponent_str = ''.join(
        map(str, primitives[2 : 2 + self.num_exponent_digits])
    )
    abs_exponent = int(abs_exponent_str, base=self.base)
    exponent = exponent_sign * abs_exponent

    mantissa_str = ''.join(map(str, primitives[2 + self.num_exponent_digits :]))
    mantissa_unscaled = int(mantissa_str, base=self.base)
    mantissa = mantissa_unscaled / self.base ** (self.num_mantissa_digits - 1)

    return sign * (self.base**exponent) * mantissa

  def from_token_ids(self, token_ids: list[int], /) -> list[float]:
    """Converts token ids to object."""
    token_strs = [self.itos[id] for id in token_ids]

    if len(token_strs) % self.num_tokens_per_obj != 0:
      raise ValueError("Tokens not a multiple of tokens per object.")

    decoded_objs = []
    for i in range(0, len(token_strs), self.num_tokens_per_obj):
      chunk = token_strs[i : i + self.num_tokens_per_obj]
      decoded_objs.append(self.tokenizer.from_tokens(chunk))

    return decoded_objs    


class NormalizedTokenizer(DecoderTokenizer[float]):
  """Tokenizer which supports only numbers within [0,1].

  A float `f` in [0,1] is represented by a sequence of integer tokens in a
  given base. For example, with base=10 and length=3, 0.523 is represented as
  [<5>, <2>, <3>]. This is equivalent to `5 * 10**-1 + 2 * 10**-2 + 3 * 10**-3`.

  Attributes:
    base: The base for the tokenization. The vocabulary will be digits from 0 to
      base-1.
    length: The number of tokens used to represent a single float. This
      determines the precision.
  """

  def __init__(self, base: int = 10, length: int = 3):
    if not base >= 2:
      raise ValueError(f'Base must be >= 2, got {base}')
    if not length >= 1:
      raise ValueError(f'Length must be >= 1, got {length}')
    self.base = base
    self.length = length

  @property
  def num_tokens_per_obj(self) -> int:
    return self.length

  def all_tokens(self) -> OrderedSet[str]:
    return OrderedSet([_to_token(i) for i in range(self.base)])

  def possible_next_tokens(self, prev_tokens: list[str]) -> OrderedSet[str]:
    if len(prev_tokens) >= self.length:
      raise ValueError(f'Index {len(prev_tokens)} out of bounds.')
    # For this tokenizer, any digit can appear at any position.
    return self.all_tokens()

  def to_tokens(self, f: float, /) -> list[str]:
    if not 0 <= f <= 1:
      raise ValueError(f'Input float must be between 0 and 1, got {f}')

    # Scale float and convert to an integer in the specified base.
    f_trunc = int(f * self.base**self.length)

    # Handle the edge case where f is exactly 1.0. The representation should
    # be the largest possible number, i.e., [base-1, base-1, ...].
    if f_trunc == self.base**self.length:
      f_trunc -= 1

    # Get the base representation, padded with zeros to the left.
    base_repr_str = np.base_repr(f_trunc, base=self.base).zfill(self.length)

    # Convert each digit character to a token, using the correct base.
    return [_to_token(int(b, self.base)) for b in base_repr_str]

  def from_tokens(self, tokens: list[str], /) -> float:
    if len(tokens) != self.length:
      raise ValueError(f'Expected {self.length} tokens, but got {len(tokens)}.')

    # Convert string tokens back to integer digits.
    token_ids = [int(str(_from_token(t))) for t in tokens]

    if not all(0 <= tid < self.base for tid in token_ids):
      raise ValueError(f'Token IDs {token_ids} out of range(0, {self.base})')

    # Calculate the float value from the integer digits.
    # The formula is: sum(d_i * base^(-(i+1))) for i in 0..length-1
    x = np.asarray(token_ids, dtype=np.float32)
    coeffs = np.power(
        self.base, -np.arange(1, self.length + 1), dtype=np.float32
    )
    return float(np.sum(x * coeffs))


class AddSpecialValues(DecoderTokenizer[str | float]):
  """Wraps a DecoderTokenizer to add special tokens."""

  SPECIAL_VALUE_MAP = {
      'INVALID': 'INVALID',
      'NAN': float('nan'),
      'INF': float('inf'),
      'NINF': float('-inf'),
  }

  def __init__(self, tokenizer: DecoderTokenizer):
    self._tokenizer = tokenizer
    self._special_tokens = OrderedSet(
        [_to_token(s) for s in self.SPECIAL_VALUE_MAP]
    )

  @property
  def num_tokens_per_obj(self) -> int:
    return self._tokenizer.num_tokens_per_obj

  def all_tokens(self) -> OrderedSet[str]:
    return self._tokenizer.all_tokens().union(self._special_tokens)

  def possible_next_tokens(self, prev_tokens: list[str]) -> OrderedSet[str]:
    if not prev_tokens:  # At start, all special tokens are possible
      return self._special_tokens.union(
          self._tokenizer.possible_next_tokens(prev_tokens)
      )
    # If a sequence starts with a special token, it must be repeated
    if prev_tokens[0] in self._special_tokens:
      return OrderedSet([prev_tokens[0]])
    return self._tokenizer.possible_next_tokens(prev_tokens)

  def to_tokens(self, obj: str | float, /) -> list[str]:
    string = None
    if isinstance(obj, str) and obj == 'INVALID':
      string = 'INVALID'
    elif isinstance(obj, float):
      if math.isnan(obj):
        string = 'NAN'
      elif obj == float('inf'):
        string = 'INF'
      elif obj == float('-inf'):
        string = 'NINF'

    if string:
      return [_to_token(string)] * self.num_tokens_per_obj
    return self._tokenizer.to_tokens(obj)

  def from_tokens(self, tokens: list[str], /) -> str | float:
    token = _from_token(tokens[0])
    if token in self.SPECIAL_VALUE_MAP:
      return self.SPECIAL_VALUE_MAP[token]
    return self._tokenizer.from_tokens(tokens)


class AppendPadTokenizer(DecoderTokenizer[ObjectT]):
  """Wraps a tokenizer to append a <pad> token after each object.

  This is a simple way to create a separator for multi-objective tasks. Note
  our model will mask out any <pad> tokens when training.
  """

  def __init__(
      self, tokenizer: DecoderTokenizer[ObjectT], *, pad_token: str = '<pad>'
  ):
    self.tokenizer = tokenizer
    self.pad_token = pad_token

  @property
  def num_tokens_per_obj(self) -> int:
    return self.tokenizer.num_tokens_per_obj + 1

  def all_tokens(self) -> OrderedSet[str]:
    return self.tokenizer.all_tokens()  # '<pad>' is not included.

  def possible_next_tokens(self, prev_tokens: list[str]) -> OrderedSet[str]:
    if len(prev_tokens) >= self.num_tokens_per_obj:
      raise ValueError(f'Index {len(prev_tokens)} out of bounds.')
    if len(prev_tokens) == self.num_tokens_per_obj - 1:  # Last position.
      return OrderedSet([self.pad_token])
    return self.tokenizer.possible_next_tokens(prev_tokens)  # Normal case.

  def to_tokens(self, obj: ObjectT, /) -> list[str]:
    return self.tokenizer.to_tokens(obj) + [self.pad_token]

  def from_tokens(self, tokens: list[str], /) -> ObjectT:
    if not tokens or tokens[-1] != self.pad_token:
      raise ValueError(f'Expected a "{self.pad_token}" token at the end.')
    return self.tokenizer.from_tokens(tokens[:-1])


class MyDigitTokenizer:

    def __init__(self,
                 vocabulary,
                 decimal_precision=4):

        self.vocabulary = vocabulary
        self.size = len(vocabulary)
        self.decimal_precision = decimal_precision
        #self.pad_id = self.size
        self.vocabulary_mapping = {v: i for i, v in enumerate(vocabulary)}
        self.ids_mapping = {i: v for v, i in self.vocabulary_mapping.items()}

    @property
    def vocab_size(self) -> int:
      return len(self.all_tokens()) + 1

    def all_tokens(self) -> OrderedSet[str]:
      tokens = self.vocabulary
      return OrderedSet([_to_token(s) for s in tokens])

    def to_tokens(self, float_lists, pad=True, drop_last=False):

        texts = utils.to_string_lists(float_lists, self.decimal_precision)

        if (drop_last):
            token_ids = [[self.vocabulary_mapping[c] for c in list(text)[:-1]] for text in texts]
        else:
            token_ids = [[self.vocabulary_mapping[c] for c in list(text)] for text in texts]

        mask = None

        if (pad):
            max_len = max([len(t) for t in token_ids])

            for i in range(len(token_ids)):
                token_ids[i] = torch.LongTensor(token_ids[i] + (max_len - len(token_ids[i]))*[self.size])
        
            token_ids = torch.stack(token_ids)
            mask =  token_ids == self.size

        return token_ids, mask

    def from_tokens(self, token_ids):
        token_ids = np.array(token_ids.cpu().numpy())
        texts = [''.join([self.ids_mapping.get(t, '') for t in tokens]) for tokens in token_ids]
        return texts

    def from_tokens_parallel(self, token_ids):
        # Work directly with tensor, no numpy conversion
        texts = []
        num_rows = token_ids.shape[0]

        # Create a lookup list for vectorized access (max ID + 1 to cover all possible IDs)
        max_id = token_ids.max().item() if token_ids.numel() > 0 else -1
        lookup_size = max(max_id + 1, len(self.ids_mapping))
        lookup_list = [self.ids_mapping.get(i, '') for i in range(lookup_size)]

        # Process every 10 rows in batches
        for i in range(0, num_rows, 10):
            # Get the next 10 rows (or remaining rows if less than 10)
            row_group = token_ids[i:i+10]
            num_rows_in_group = row_group.shape[0]

            # Flatten all IDs in the group for parallel processing
            flat_ids = row_group.flatten().cpu().tolist()

            # Vectorized mapping: map all IDs to characters in parallel
            # Direct indexing with bounds check for safety
            mapped_chars = [lookup_list[idx] if 0 <= idx < lookup_size else '' for idx in flat_ids]
            
            # Reshape back to rows and join characters into strings
            seq_len = row_group.shape[1] if len(row_group.shape) > 1 else len(mapped_chars) // num_rows_in_group
            group_texts = []
            for row_idx in range(num_rows_in_group):
                start_idx = row_idx * seq_len
                end_idx = start_idx + seq_len
                row_chars = mapped_chars[start_idx:end_idx]
                group_texts.append(''.join(row_chars))
            
            texts.append(group_texts)
        
        return texts      

class MyIEEETokenizer:
    """
    MyIEEETokenizer

    Wrapper around P10Tokenizer that exposes an interface similar to MyDigitTokenizer.

    - Uses P10-style encoding: each float is represented as:
        [sign_token, digit_1, ..., digit_{num_digits}, exponent_token]
    - Tokens are those produced by P10Tokenizer (e.g. '<+>', '<3>', '<E-10>', ...).
    - We maintain an integer vocabulary over these tokens + an implicit pad id.

    Args:
      num_digits: number of mantissa digits used in P10Tokenizer.
      exponent_range: exponent range in P10Tokenizer ([-range, +range]).

    Methods:
      to_tokens(float_lists, pad=True, drop_last=False)
        float_lists : list of list of floats (batch of sequences)
        Returns:
          token_ids : LongTensor of shape (B, T) if pad=True, else list[LongTensor]
          mask      : BoolTensor of shape (B, T) with True where pad, else None

      from_tokens(token_ids)
        token_ids : LongTensor of shape (B, T)
        Returns:
          decoded : list of list of floats (one list per batch element)
    """

    def __init__(self,
                 num_digits: int = 4,
                 exponent_range: int = 10):
        # Underlying P10-style tokenizer for single floats
        self.p10 = P10Tokenizer(num_digits=num_digits,
                                exponent_range=exponent_range)

        # Vocabulary = all P10 tokens (already in <...> form)
        self.vocabulary = list(self.p10.all_tokens())
        self.size = len(self.vocabulary)        # last id is reserved as pad_id

        # Mappings token <-> id
        self.vocabulary_mapping = {v: i for i, v in enumerate(self.vocabulary)}
        self.ids_mapping = {i: v for v, i in self.vocabulary_mapping.items()}

        # Implicit pad id (just like in MyDigitTokenizer)
        self.pad_id = self.size

        # For reference
        self.num_digits = num_digits
        self.exponent_range = exponent_range

    @property
    def vocab_size(self) -> int:
        # All real tokens + 1 pad token
        return len(self.all_tokens()) + 1

    def all_tokens(self) -> OrderedSet[str]:
        # Delegates to underlying P10 tokenizer
        return self.p10.all_tokens()

    # -------------------------------------------------------------------------
    # Encoding
    # -------------------------------------------------------------------------
    def _encode_sequence(self,
                         floats: list[float],
                         drop_last: bool = False) -> list[int]:
        """
        Encode a single list of floats into a flat list of token ids.
        """
        token_strs: list[str] = []
        for f in floats:
            # P10Tokenizer.to_tokens returns a list[str] of serialized tokens
            token_strs.extend(self.p10.to_tokens(f))

        if drop_last and token_strs:
            token_strs = token_strs[:-1]

        # Map tokens to ids
        token_ids = [self.vocabulary_mapping[t] for t in token_strs]
        return token_ids

    def to_tokens(self,
                  float_lists: list[list[float]],
                  pad: bool = True,
                  drop_last: bool = False):
        """
        Encode a batch of float sequences.

        Args:
          float_lists: list of list of floats, shape (B, variable_length).
          pad: whether to left-pad sequences to a common length.
          drop_last: if True, drop the last token of each encoded sequence.

        Returns:
          token_ids:
            - if pad=True: LongTensor of shape (B, T_max)
            - if pad=False: list[LongTensor] (ragged)
          mask:
            - if pad=True: BoolTensor (B, T_max) where True = pad position
            - if pad=False: None
        """
        # Encode each sequence individually
        encoded_seqs = [
            self._encode_sequence(floats, drop_last=drop_last)
            for floats in float_lists
        ]

        mask = None

        if pad:
            max_len = max(len(seq) for seq in encoded_seqs) if encoded_seqs else 0

            padded_seqs = []
            for seq in encoded_seqs:
                pad_len = max_len - len(seq)
                padded = seq + [self.pad_id] * pad_len
                padded_seqs.append(torch.LongTensor(padded))

            token_ids = torch.stack(padded_seqs)
            mask = token_ids == self.pad_id
        else:
            token_ids = [torch.LongTensor(seq) for seq in encoded_seqs]

        return token_ids, mask

    # -------------------------------------------------------------------------
    # Decoding
    # -------------------------------------------------------------------------
    def from_tokens(self, token_ids):
        """
        Decode a batch of token id sequences back into floats.

        Args:
          token_ids: torch.LongTensor of shape (B, T) OR equivalent
                     (like a numpy array / tensor on CPU).

        Returns:
          decoded: list[list[float]]
            decoded[b] is the list of floats for the b-th sequence.
        """
        # Convert to numpy for easier iteration (assumes on CPU)
        token_ids = np.array(token_ids.cpu().numpy())
        if token_ids.ndim == 1:
            # Promote to batch dimension if a single sequence is passed
            token_ids = token_ids[None, :]

        decoded_batch: list[list[float]] = []
        tokens_per_obj = self.p10.num_tokens_per_obj

        for row in token_ids:
            # Strip pad positions and map ids -> token strings
            valid_ids = [int(t) for t in row if t != self.pad_id]
            token_strs = [self.ids_mapping[i] for i in valid_ids]

            if len(token_strs) % tokens_per_obj != 0:
                raise ValueError(
                    f"Token length {len(token_strs)} is not a multiple of "
                    f"tokens_per_obj={tokens_per_obj}.")

            floats: list[float] = []
            for i in range(0, len(token_strs), tokens_per_obj):
                chunk = token_strs[i:i + tokens_per_obj]
                f = self.p10.from_tokens(chunk)
                floats.append(f)

            decoded_batch.append(floats)

        return decoded_batch