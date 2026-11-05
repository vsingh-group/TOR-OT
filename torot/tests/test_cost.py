import itertools

import numpy as np
import pytest
from datasets import ClassLabel, Dataset, Features, Value

from torot.data.cost import COST_TYPES, cost_matrix
from torot.data.dataset import cost_label_map
from torot.data.supports import grid_1d


def test_costs_are_distinct_symmetric_and_normalised_in_1d():
    xs = grid_1d(30)
    mats = {c: cost_matrix(xs, xs, c) for c in COST_TYPES}
    for c, C in mats.items():
        assert C.shape == (30, 30)
        assert np.allclose(C, C.T), c
        assert np.isclose(C.max(), 1.0) and C.min() >= 0.0, c
        assert np.allclose(np.diag(C), 0.0), c
    for a, b in itertools.combinations(COST_TYPES, 2):
        assert not np.allclose(mats[a], mats[b]), f"{a} and {b} coincide in 1D"


def test_cost_label_map_accepts_matching_and_rejects_stale_corpus():
    feats = Features({"cost_type": ClassLabel(names=list(reversed(COST_TYPES))), "n": Value("int32")})
    ds = Dataset.from_dict({"cost_type": [0, 1], "n": [5, 5]}, features=feats)
    mapping = cost_label_map(ds)
    assert [COST_TYPES[i] for i in mapping] == list(reversed(COST_TYPES))

    stale = Features({"cost_type": ClassLabel(names=["sqeuclidean", "cityblock"]), "n": Value("int32")})
    with pytest.raises(ValueError, match="cityblock"):
        cost_label_map(Dataset.from_dict({"cost_type": [0], "n": [5]}, features=stale))
