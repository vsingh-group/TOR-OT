"""1D coordinates may be given as one column (x), a flat array, or padded (x, 0): the model sees the same input."""
import numpy as np
import torch

from torot.data.cost import COST_TYPES
from torot.data.dataset import Collator
from torot.data.supports import grid_1d, pad_coords
from torot.decode import predict_plans
from torot.tests.test_row_decoding import tiny_row_model
from torot.tokenizer import PlanTokenizer


def instance(coords, n=7, seed=0):
    rng = np.random.default_rng(seed)
    return {"n": n, "m": n, "dim": 1, "cost_type": COST_TYPES.index("sqeuclidean"),
            "coords_src": coords, "coords_tgt": coords, "a": rng.dirichlet(np.ones(n)), "b": rng.dirichlet(np.ones(n))}


def test_collator_pads_1d_coordinates_with_zeros():
    tok, x = PlanTokenizer(), grid_1d(7)                       # shape [7, 1]
    batches = [Collator(tok)([instance(c)]) for c in (x, x[:, 0], pad_coords(x))]
    for batch in batches:
        assert torch.equal(batch["coords_src"], batches[2]["coords_src"])
        assert torch.equal(batch["coords_tgt"], batches[2]["coords_tgt"])
    assert torch.all(batches[0]["coords_src"][..., 1] == 0)


def test_predicted_plan_does_not_depend_on_coordinate_padding():
    torch.manual_seed(0)
    tok = PlanTokenizer()
    model = tiny_row_model(tok).eval()
    x = grid_1d(7)
    plans = [predict_plans(model, tok, [instance(c)], torch.device("cpu"), max_len=256, last_row="model")[0]
             for c in (x, pad_coords(x))]
    assert np.array_equal(plans[0], plans[1])
