import numpy as np
import pytest

from torot.tokenizer import KIND_DIGIT, KIND_EXP, KIND_STRUCT, KIND_TGT, GrammarState, PlanTokenizer


def random_plan(rng, n, m, nnz):
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, m, size=nnz)
    vals = 10.0 ** rng.uniform(-6, 0, size=nnz)
    # drop duplicate (row, col) pairs to keep the plan a valid COO
    _, keep = np.unique(rows * m + cols, return_index=True)
    return rows[keep], cols[keep], vals[keep]


@pytest.mark.parametrize("v", [1.0, 0.5, 0.03127, 0.099995, 0.00012345, 1e-9, 0.99996])
def test_mass_round_trip_relative_error(v):
    tok = PlanTokenizer(mantissa_digits=4, max_exponent=9)
    k, digits = tok.encode_mass(v)
    assert 1 <= k <= tok.max_exponent
    assert digits[0] != "0"
    dec = tok.decode_mass(k, [int(c) for c in digits])
    assert abs(dec - v) / v < 1e-3


def test_mass_below_range_is_dropped():
    tok = PlanTokenizer(mantissa_digits=4, max_exponent=6)
    assert tok.encode_mass(1e-8) is None


def test_plan_round_trip_and_kinds():
    rng = np.random.default_rng(0)
    tok = PlanTokenizer()
    n, m = 12, 9
    rows, cols, vals = random_plan(rng, n, m, 25)
    ids, kinds = tok.encode(rows, cols, vals, n)
    assert ids[0] == tok.BOS and ids[-1] == tok.EOS
    assert ids.count(tok.ROW) == n                      # exactly one ROW block per source bin
    assert kinds[0] == KIND_STRUCT and kinds[-1] == KIND_STRUCT
    assert set(kinds) <= {KIND_STRUCT, KIND_TGT, KIND_EXP} | {KIND_DIGIT + p for p in range(tok.D)}

    r, c, v = tok.decode(ids)
    order = np.lexsort((cols, rows))
    assert np.array_equal(r, rows[order])
    assert np.array_equal(c, cols[order])
    assert np.all(np.abs(v - vals[order]) / vals[order] < 1e-3)


def test_empty_rows_round_trip():
    tok = PlanTokenizer()
    rows, cols, vals = [1, 3], [0, 2], [0.5, 0.5]
    ids, _ = tok.encode(rows, cols, vals, n=5)          # rows 0, 2 and 4 are empty
    assert ids.count(tok.ROW) == 5
    r, c, _ = tok.decode(ids)
    assert r.tolist() == rows and c.tolist() == cols


def test_grammar_mask_admits_reference_sequence():
    rng = np.random.default_rng(1)
    tok = PlanTokenizer()
    for _ in range(20):
        n, m = int(rng.integers(1, 20)), int(rng.integers(1, 20))
        m_pad = m + int(rng.integers(0, 4))
        rows, cols, vals = random_plan(rng, n, m, int(rng.integers(1, 30)))
        ids, _ = tok.encode(rows, cols, vals, n)
        state = GrammarState()
        for t in ids[1:]:
            mask = tok.allowed_mask(state, n, m, m_pad)
            assert mask[t], f"token {t} rejected in phase {state.phase} (row {state.row})"
            tok.advance(state, t)
        assert state.finished


def test_empty_plan_encodes_to_n_rows():
    tok = PlanTokenizer()
    ids, _ = tok.encode([], [], [], n=3)
    assert ids == [tok.BOS, tok.ROW, tok.ROW, tok.ROW, tok.EOS]
