import math

import numpy as np
from TF4OT import tokenizers

from absl.testing import absltest
from absl.testing import parameterized

class P10TokenizerTest(parameterized.TestCase):

  @parameterized.parameters(
      (123.4, '<+><1><2><3><4><E-1>', 123.4),
      (12345, '<+><1><2><3><4><E1>', 12340),
      (0.1234, '<+><1><2><3><4><E-4>', 0.1234),
      (-123.4, '<-><1><2><3><4><E-1>', -123.4),
      (-12345, '<-><1><2><3><4><E1>', -12340),
      (-0.1234, '<-><1><2><3><4><E-4>', -0.1234),
      (0.0, '<+><0><0><0><0><E0>', 0.0),
      (-0.0, '<+><0><0><0><0><E0>', 0.0),  # in python, 0.0 == -0.0
      (-0.4e-13, '<-><0><0><0><0><E0>', 0.0),  # note leading negative zero
  )
  def test_tokenize(self, f: float, tokens: str, f_prime: float):
    vocab = tokenizers.P10Tokenizer()
    out_tokens = vocab.to_tokens(f)
    self.assertEqual(''.join(out_tokens), tokens)
    self.assertAlmostEqual(vocab.from_tokens(out_tokens), f_prime)

  @parameterized.parameters((3, 10, 1.0e-8, 9.99e12), (1, 5, 1.0e-5, 9.0e5))
  def test_representation_range(
      self,
      num_digits: int,
      exponent_range: int,
      min_val: float,
      max_val: float,
  ):
    vocab = tokenizers.P10Tokenizer(
        num_digits=num_digits,
        exponent_range=exponent_range,
    )
    self.assertEqual(vocab._max_abs_val, max_val)
    self.assertEqual(vocab._min_abs_val, min_val)

  @parameterized.parameters(
      (1.0e13, 3, 10, '<+><9><9><9><E10>'),
      (2.0e13, 3, 10, '<+><9><9><9><E10>'),
      (-1.0e13, 3, 10, '<-><9><9><9><E10>'),
      (-2.0e13, 3, 10, '<-><9><9><9><E10>'),
      (9.9e12, 3, 10, '<+><9><9><0><E10>'),
      (-9.9e12, 3, 10, '<-><9><9><0><E10>'),
      (5.0e5, 3, 10, '<+><5><0><0><E3>'),
      (1.1e-8, 3, 10, '<+><1><1><0><E-10>'),
      (0.9e-8, 3, 10, '<+><1><0><0><E-10>'),
      (0.5e-8, 3, 10, '<+><0><0><0><E0>'),
      (0.51e-8, 3, 10, '<+><1><0><0><E-10>'),
      (0.4e-8, 3, 10, '<+><0><0><0><E0>'),
      # rounding up below creats a negative sign for 0
      (-0.4e-8, 3, 10, '<-><0><0><0><E0>'),
      (-0.5e-8, 3, 10, '<-><0><0><0><E0>'),
      (-0.51e-8, 3, 10, '<-><1><0><0><E-10>'),
      (-0.8e-8, 3, 10, '<-><1><0><0><E-10>'),
      (-1.1e-8, 3, 10, '<-><1><1><0><E-10>'),
  )
  def test_tokenization_limit(
      self,
      f: float,
      num_digits: int,
      exponent_range: int,
      serialized: str,
  ):
    vocab = tokenizers.P10Tokenizer(
        num_digits=num_digits,
        exponent_range=exponent_range,
    )
    self.assertEqual(''.join(vocab.to_tokens(f)), serialized)

  def test_all_tokens_used(self):
    vocab = tokenizers.P10Tokenizer(exponent_range=2)
    out = vocab.all_tokens()

    signs = ['<+>', '<->']
    digits = [f'<{i}>' for i in range(10)]
    exponents = ['<E-2>', '<E-1>', '<E0>', '<E1>', '<E2>']
    self.assertEqual(list(out), signs + digits + exponents)


class IEEEFloatTokenizerTest(parameterized.TestCase):

  @parameterized.parameters(
      (123.4, '<+><+><2><1><2><3><4>', 123.4),
      (12345, '<+><+><4><1><2><3><4>', 12340),
      (0.1234, '<+><-><1><1><2><3><4>', 0.1234),
      (-123.4, '<-><+><2><1><2><3><4>', -123.4),
      (-12345, '<-><+><4><1><2><3><4>', -12340),
      (-0.1234, '<-><-><1><1><2><3><4>', -0.1234),
      (1.234e-9, '<+><-><9><1><2><3><4>', 1.234e-9),
      (-1.234e-9, '<-><-><9><1><2><3><4>', 1.234e-9),
      (1.2e-9, '<+><-><9><1><2><0><0>', 1.2e-9),
      (-1.2e-9, '<-><-><9><1><2><0><0>', -1.2e-9),
      (1.2e9, '<+><+><9><1><2><0><0>', 1.2e9),
      (-1.2e9, '<-><+><9><1><2><0><0>', -1.2e9),
      (1.2345e9, '<+><+><9><1><2><3><4>', 1.234e9),
      (1.234e-10, '<+><-><0><0><0><0><0>', 0.0),  # Underflow
      (-1.234e-10, '<-><-><0><0><0><0><0>', 0.0),  # Underflow
      (0.0, '<+><+><0><0><0><0><0>', 0.0),
      (-0.0, '<+><+><0><0><0><0><0>', 0.0),  # in python, 0.0 == -0.0
  )
  def test_tokenize(self, f: float, tokens: str, f_prime: float):
    vocab = tokenizers.IEEEFloatTokenizer()
    out_tokens = vocab.to_tokens(f)
    self.assertEqual(''.join(out_tokens), tokens)
    self.assertAlmostEqual(vocab.from_tokens(out_tokens), f_prime)

  @parameterized.parameters((3,), (10,), (18,))
  def test_all_tokens_used(self, base: int):
    vocab = tokenizers.IEEEFloatTokenizer(base=base)
    out = vocab.all_tokens()

    signs = ['<+>', '<->']
    digits = [f'<{i}>' for i in range(base)]
    self.assertEqual(list(out), signs + digits)


class NormalizedTokenizerTest(parameterized.TestCase):

  @parameterized.parameters(
      # Base 10 tests
      (0.123, 10, 3, '<1><2><3>', 0.123),
      (0.0, 10, 3, '<0><0><0>', 0.0),
      (1.0, 10, 3, '<9><9><9>', 0.999),
      (0.999, 10, 3, '<9><9><9>', 0.999),
      (0.1, 10, 3, '<1><0><0>', 0.1),
      (0.12345, 10, 4, '<1><2><3><4>', 0.1234),  # Truncation
      # Base 2 tests
      (0.5, 2, 3, '<1><0><0>', 0.5),  # 1/2
      (0.25, 2, 3, '<0><1><0>', 0.25),  # 1/4
      (0.875, 2, 3, '<1><1><1>', 0.875),  # 1/2 + 1/4 + 1/8
      (1.0, 2, 3, '<1><1><1>', 0.875),
      # Base 16 tests
      (0.5, 16, 2, '<8><0>', 0.5),  # 8/16
      (1.0, 16, 2, '<15><15>', 15 / 16 + 15 / 256),
  )
  def test_tokenize_and_from_tokens(
      self, f: float, base: int, length: int, tokens_str: str, f_prime: float
  ):
    tokenizer = tokenizers.NormalizedTokenizer(base=base, length=length)

    # Test tokenization
    out_tokens = tokenizer.to_tokens(f)
    self.assertEqual(''.join(out_tokens), tokens_str)

    # Test de-tokenization
    reconstructed_f = tokenizer.from_tokens(out_tokens)
    self.assertTrue(
        np.isclose(reconstructed_f, f_prime),
        msg=f'Expected {f_prime}, but got {reconstructed_f}',
    )

  @parameterized.parameters((2,), (10,), (16,))
  def test_all_tokens_used(self, base: int):
    tokenizer = tokenizers.NormalizedTokenizer(base=base)
    all_tokens = tokenizer.all_tokens()
    expected_tokens = [f'<{i}>' for i in range(base)]
    self.assertEqual(list(all_tokens), expected_tokens)

  def test_possible_next_tokens(self):
    tokenizer = tokenizers.NormalizedTokenizer(base=10, length=4)
    expected_tokens = tokenizer.all_tokens()

    # At the beginning
    self.assertEqual(tokenizer.possible_next_tokens([]), expected_tokens)
    # In the middle
    self.assertEqual(
        tokenizer.possible_next_tokens(['<1>', '<2>']), expected_tokens
    )

    # Test bounds check
    with self.assertRaises(ValueError):
      tokenizer.possible_next_tokens(['<1>', '<2>', '<3>', '<4>'])

  def test_input_validation(self):
    tokenizer = tokenizers.NormalizedTokenizer(base=10, length=3)

    # to_tokens input range
    with self.assertRaisesRegex(ValueError, 'must be between 0 and 1'):
      tokenizer.to_tokens(1.1)
    with self.assertRaisesRegex(ValueError, 'must be between 0 and 1'):
      tokenizer.to_tokens(-0.1)

    # from_tokens length mismatch
    with self.assertRaisesRegex(ValueError, 'Expected 3 tokens, but got 2'):
      tokenizer.from_tokens(['<1>', '<2>'])

    # from_tokens out-of-range token
    with self.assertRaisesRegex(ValueError, 'out of range'):
      tokenizer.from_tokens(['<1>', '<10>', '<3>'])


class AddSpecialValuesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.base_tokenizer = tokenizers.P10Tokenizer(
        num_digits=1, exponent_range=2
    )
    self.tokenizer = tokenizers.AddSpecialValues(self.base_tokenizer)

  @parameterized.parameters(
      ('INVALID', ['<INVALID>', '<INVALID>', '<INVALID>']),
      (float('nan'), ['<NAN>', '<NAN>', '<NAN>']),
      (float('inf'), ['<INF>', '<INF>', '<INF>']),
      (float('-inf'), ['<NINF>', '<NINF>', '<NINF>']),
      (100.0, ['<+>', '<1>', '<E2>']),
  )
  def test_tokenize(self, f: float, expected_tokens: list[str]):
    out_tokens = self.tokenizer.to_tokens(f)
    self.assertEqual(out_tokens, expected_tokens)
    f_prime = self.tokenizer.from_tokens(out_tokens)
    if f_prime == 'INVALID':
      self.assertEqual(f_prime, 'INVALID')
    elif math.isnan(f):
      self.assertTrue(math.isnan(f_prime))
    else:
      self.assertAlmostEqual(f_prime, f)

  def test_all_tokens_used(self):
    self.assertEqual(
        list(self.tokenizer.all_tokens()),
        ['<+>', '<->']
        + [f'<{i}>' for i in range(10)]
        + ['<E-2>', '<E-1>', '<E0>', '<E1>', '<E2>']
        + ['<INVALID>', '<NAN>', '<INF>', '<NINF>'],
    )

  def test_possible_next_tokens(self):
    self.assertEqual(
        self.tokenizer.possible_next_tokens([]),
        ['<INVALID>', '<NAN>', '<INF>', '<NINF>'] + ['<+>', '<->'],
    )

    self.assertEqual(
        self.tokenizer.possible_next_tokens(['<+>']),
        [f'<{i}>' for i in range(10)],
    )

    self.assertEqual(
        self.tokenizer.possible_next_tokens(['<INVALID>']), ['<INVALID>']
    )


class AppendPadTokenizerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Use a simple base tokenizer for clarity
    self.base_tokenizer = tokenizers.P10Tokenizer(
        num_digits=2, exponent_range=1
    )
    self.tokenizer = tokenizers.AppendPadTokenizer(self.base_tokenizer)

  def test_to_tokens_appends_pad(self):
    wrapped_tokens = self.tokenizer.to_tokens(12.0)
    self.assertEqual(wrapped_tokens, ['<+>', '<1>', '<2>', '<E0>', '<pad>'])

  def test_from_tokens_strips_pad(self):
    tokens_with_pad = ['<+>', '<1>', '<2>', '<E0>', '<pad>']
    self.assertAlmostEqual(self.tokenizer.from_tokens(tokens_with_pad), 12.0)

    # Fails if the pad token is missing
    tokens_without_pad = ['<+>', '<1>', '<2>', '<E0>']
    with self.assertRaisesRegex(ValueError, 'Expected a "<pad>" token'):
      self.tokenizer.from_tokens(tokens_without_pad)

  def test_possible_next_tokens(self):
    # No change at start or middle.
    self.assertEqual(
        self.tokenizer.possible_next_tokens([]),
        self.base_tokenizer.possible_next_tokens([]),
    )
    self.assertEqual(
        self.tokenizer.possible_next_tokens(['<+>']),
        self.base_tokenizer.possible_next_tokens(['<+>']),
    )

    # Final position should only be pad.
    end_of_base_tokens = ['<+>', '<1>', '<2>', '<E1>']
    self.assertEqual(
        self.tokenizer.possible_next_tokens(end_of_base_tokens),
        tokenizers.OrderedSet(['<pad>']),
    )

    # Check out of bounds.
    with self.assertRaises(ValueError):
      self.tokenizer.possible_next_tokens(end_of_base_tokens + ['<pad>'])


class DigitTokenizerTest(parameterized.TestCase):

  @parameterized.parameters(
      # (alphabet, length, pad, input_str, serialized_tokens, reconstructed_str)
      (['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', '.', '[', ']', ' '], 5, 'PAD', '[0,0.075,1,0.032]', '<1><2><3><PAD><PAD>', '123'),
      ('0123456789', 5, 'PAD', '123', '<1><2><3><PAD><PAD>', '123'),
      ('0123456789', 5, 'PAD', '', '<PAD><PAD><PAD><PAD><PAD>', ''),
      ('0123456789', 4, 'PAD', '0075', '<0><0><7><5>', '0075'),
      ('abc', 4, '_', 'ab', '<a><b><_><_>', 'ab'),
      ('0123456789.-+', 6, 'PAD', '-1.2', '<-><1><.><2><PAD><PAD>', '-1.2'),
  )
  def test_tokenize_and_roundtrip(
      self, alphabet, length, pad, s, tokens_str, s_prime
  ):
    tok = tokenizers.DigitTokenizer(
        alphabet=alphabet, length=length, pad_symbol=pad
    )
    out_tokens = tok.to_tokens(s)
    self.assertEqual(''.join(out_tokens), tokens_str)
    self.assertEqual(tok.from_tokens(out_tokens), s_prime)

  def test_all_tokens_used(self):
    tok = tokenizers.DigitTokenizer(
        alphabet=['0', '1', 'X'], length=3, pad_symbol='PAD'
    )
    # all_tokens returns an OrderedSet of serialized strings
    expected = ['<0>', '<1>', '<X>', '<PAD>']
    self.assertEqual(list(tok.all_tokens()), expected)

  def test_num_tokens_per_obj(self):
    tok = tokenizers.DigitTokenizer(alphabet='01', length=4, pad_symbol='P')
    self.assertEqual(tok.num_tokens_per_obj, 4)

  def test_possible_next_tokens(self):
    tok = tokenizers.DigitTokenizer(alphabet='01', length=3, pad_symbol='P')
    expected = tok.all_tokens()

    # At the start and in the middle, any symbol (including pad) is allowed.
    self.assertEqual(tok.possible_next_tokens([]), expected)
    self.assertEqual(tok.possible_next_tokens(['<0>']), expected)

    # Bounds check: exactly length prev_tokens -> out of bounds
    with self.assertRaises(ValueError):
      tok.possible_next_tokens(['<0>', '<1>', '<P>'])

  def test_length_enforcement(self):
    tok = tokenizers.DigitTokenizer(alphabet='0123456789', length=3, pad_symbol='PAD')
    with self.assertRaisesRegex(ValueError, 'exceeds fixed length'):
      tok.to_tokens('1234')

  def test_invalid_characters(self):
    tok = tokenizers.DigitTokenizer(alphabet='0123456789', length=3, pad_symbol='PAD')
    with self.assertRaisesRegex(ValueError, 'not in alphabet/pad set'):
      tok.to_tokens('1A3')  # 'A' not in alphabet

  def test_from_tokens_length_mismatch(self):
    tok = tokenizers.DigitTokenizer(alphabet='0123456789', length=3, pad_symbol='PAD')
    with self.assertRaisesRegex(ValueError, 'Expected 3 tokens'):
      tok.from_tokens(['<1>', '<2>'])

  def test_from_tokens_invalid_symbol(self):
    tok = tokenizers.DigitTokenizer(alphabet='0123456789', length=3, pad_symbol='PAD')
    # '<Z>' is not part of the tokenizer's token set
    with self.assertRaisesRegex(ValueError, 'not in alphabet/pad set'):
      tok.from_tokens(['<1>', '<Z>', '<PAD>'])

  def test_trailing_pad_stripped_on_decode(self):
    tok = tokenizers.DigitTokenizer(alphabet='0123456789', length=5, pad_symbol='PAD')
    # Manually craft tokens with pads; decode should strip trailing PADs.
    tokens_with_pad = ['<0>', '<7>', '<5>', '<PAD>', '<PAD>']
    self.assertEqual(tok.from_tokens(tokens_with_pad), '075')

if __name__ == '__main__':
  absltest.main()