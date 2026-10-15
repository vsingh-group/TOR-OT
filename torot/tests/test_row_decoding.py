import numpy as np
import ot
import torch
from datasets import Dataset

from torot.config import LossConfig, ModelConfig
from torot.data.cost import COST_TYPES, cost_matrix
from torot.data.dataset import Collator, OTRowDataset, dense_plan, record_to_instance
from torot.data.generate import GenConfig, make_random_instance
from torot.data.schema import FEATURES
from torot.decode import decode_by_rows, predict_plans
from torot.loss import PlanLoss
from torot.model import ToROT
from torot.tokenizer import PlanTokenizer


def small_split(num=12, cost_types=None, seed0=0):
    cfg = GenConfig(n_min=5, n_max=9, cost_types=cost_types or list(COST_TYPES))
    recs = [make_random_instance(seed0 + s, cfg) for s in range(num)]
    return Dataset.from_list([r for r in recs if r is not None], features=FEATURES)


def tiny_row_model(tok):
    cfg = ModelConfig(d_model=64, n_heads=4, enc_layers=1, dec_layers=2, d_ff=128, fourier_bands=4,
                      max_seq_len=256, decoding="row")
    return ToROT(cfg, tok.vocab_size)


def test_row_tokens_round_trip_and_grammar():
    tok = PlanTokenizer()
    cols, vals = np.array([1, 4, 7]), np.array([0.25, 0.0031, 0.1])
    ids, kinds = tok.encode_row(cols, vals)
    assert ids[0] == tok.BOS and ids[-1] == tok.EOS and tok.ROW not in ids
    assert len(ids) == len(kinds) == 2 + 3 * (tok.D + 2)
    c, v = tok.decode_row(ids)
    assert c.tolist() == cols.tolist() and np.allclose(v, vals, rtol=1e-3)
    state = tok.row_state()
    for t in ids[1:]:
        assert tok.allowed_mask(state, 1, 9, 10)[t]
        tok.advance(state, t)
    assert state.finished


def test_row_dataset_examples_are_remaining_problems():
    split = small_split()
    ds = OTRowDataset(split)
    assert len(ds) == sum(n - 1 for n in split["n"])
    sizes = ds.sizes()
    for idx in range(len(ds)):
        k, i = ds.locate(idx)
        inst = record_to_instance(split[k], ds.cost_map)
        P = dense_plan(inst)
        ex = ds[idx]
        assert ex["n"] == inst["n"] - i and sizes[idx] == ex["n"] + ex["m"]
        assert np.allclose(ex["a"], inst["a"][i:])
        assert np.allclose(ex["b"], np.maximum(inst["b"] - P[:i].sum(0), 0.0))
        assert np.allclose(P[i, ex["row_cols"]], ex["row_vals"]) and np.isclose(ex["row_vals"].sum(), inst["a"][i])


def test_row_is_first_row_of_remaining_optimal_plan():
    """The property row decoding relies on, checked with POT on costs with a unique optimum."""
    split = small_split(num=8, cost_types=["sqeuclidean", "cubic_euclidean"])
    ds = OTRowDataset(split)
    for idx in range(len(ds)):
        ex = ds[idx]
        C = cost_matrix(ex["coords_src"][:, :1], ex["coords_tgt"][:, :1], COST_TYPES[ex["cost_type"]])
        total = ex["a"].sum()
        sub = ot.emd(ex["a"] / total, ex["b"] / ex["b"].sum(), C) * total
        target = np.zeros(ex["m"]); target[ex["row_cols"]] = ex["row_vals"]
        assert np.allclose(sub[0], target, atol=1e-9)


def test_row_model_trains_and_decodes():
    tok = PlanTokenizer()
    split = small_split(num=6)
    ds = OTRowDataset(split)
    model = tiny_row_model(tok)
    batch = Collator(tok)([ds[i] for i in range(8)])
    logits = model(batch)
    assert logits.shape[:2] == batch["input_ids"].shape
    loss = PlanLoss(tok, **vars(LossConfig()))(logits, batch["target_ids"], batch["kinds"])
    assert torch.isfinite(loss)
    loss.backward()

    model.eval()
    instances = [record_to_instance(split[k], ds.cost_map) for k in range(3)]
    plans = decode_by_rows(model, tok, instances, torch.device("cpu"), max_len=200)
    for P, inst in zip(plans, instances):
        assert P.shape == (inst["n"], inst["m"]) and (P >= 0).all()
        # columns can only be over-filled by earlier rows; the last row takes exactly what remains
        assert np.all(P.sum(0) >= inst["b"] - 1e-9)
    sym = predict_plans(model, tok, instances[:1], torch.device("cpu"), max_len=200, symmetric=True)
    assert sym[0].shape == plans[0].shape


def test_leftover_augmentation_only_touches_used_up_columns():
    split = small_split(num=10)
    exact = OTRowDataset(split)
    aug = OTRowDataset(split, residual_leftover_prob=1.0, residual_leftover_range=(1e-6, 1e-2))
    seen_leftover = False
    for idx in range(len(exact)):
        e, a = exact[idx], aug[idx]
        used_up = e["b"] <= 1e-12
        assert np.allclose(a["b"][~used_up], e["b"][~used_up])                     # other capacities untouched
        assert np.all((a["b"][used_up] >= 1e-6) & (a["b"][used_up] <= 1e-2))       # every used-up column got a leftover
        assert np.array_equal(a["row_cols"], e["row_cols"]) and np.allclose(a["row_vals"], e["row_vals"])  # target unchanged
        seen_leftover |= bool(used_up.any())
    assert seen_leftover
