import numpy as np
import ot
import torch

from torot.config import LossConfig, ModelConfig
from torot.data.cost import cost_matrix
from torot.data.dataset import Collator, split_batch, transpose_instance
from torot.data.supports import grid_1d, pad_coords
from torot.decode import generate, plan_to_dense, predict_plans
from torot.loss import PlanLoss
from torot.metrics import evaluate_instance
from torot.model import ToROT
from torot.tokenizer import PlanTokenizer


def make_instance(rng, n, m, cost_type=0):
    xs, xt = grid_1d(n), grid_1d(m)
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(m))
    C = cost_matrix(xs, xt, "sqeuclidean")
    plan = ot.emd(a, b, C)
    rows, cols = np.nonzero(plan > 1e-12)
    return {
        "n": n,
        "m": m,
        "dim": 1,
        "cost_type": cost_type,
        "coords_src": pad_coords(xs),
        "coords_tgt": pad_coords(xt),
        "a": a,
        "b": b,
        "plan_rows": rows,
        "plan_cols": cols,
        "plan_vals": plan[rows, cols],
        "ot_cost": float((plan * C).sum()),
    }


def tiny_model(tok):
    cfg = ModelConfig(d_model=64, n_heads=4, enc_layers=1, dec_layers=2, d_ff=128, fourier_bands=4, max_seq_len=512,
                      decoding="plan")
    return ToROT(cfg, tok.vocab_size)


def test_forward_shapes_and_loss():
    rng = np.random.default_rng(0)
    tok = PlanTokenizer()
    instances = [make_instance(rng, 6, 9), make_instance(rng, 8, 5)]
    batch = Collator(tok)(instances)
    model = tiny_model(tok)
    logits = model(batch)
    B, T = batch["input_ids"].shape
    assert logits.shape == (B, T, tok.extended_size(9))
    loss = PlanLoss(tok, **vars(LossConfig()))(logits, batch["target_ids"], batch["kinds"])
    assert torch.isfinite(loss)
    loss.backward()


def test_micro_batches_give_the_full_batch_gradient():
    rng = np.random.default_rng(1)
    tok = PlanTokenizer()
    instances = [make_instance(rng, n, m) for n, m in [(4, 5), (9, 7), (5, 12), (3, 3), (8, 8), (6, 4)]]
    batch = Collator(tok)(instances)
    B, T = batch["input_ids"].shape
    micro = split_batch(batch, max_tokens=2 * T)
    assert len(micro) > 1 and all(mb["input_ids"].numel() <= 2 * T for mb in micro)
    assert sum(len(mb["n"]) for mb in micro) == B

    torch.manual_seed(0)
    model = tiny_model(tok)
    loss_fn = PlanLoss(tok, **vars(LossConfig()))
    loss_fn(model(batch), batch["target_ids"], batch["kinds"]).backward()
    full = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    norms = loss_fn.normalizers(batch["kinds"])
    for mb in micro:
        loss_fn(model(mb), mb["target_ids"], mb["kinds"], norms).backward()
    for g_full, p in zip(full, model.parameters()):
        torch.testing.assert_close(p.grad, g_full, rtol=1e-4, atol=1e-6)


def test_cached_decoding_matches_full_forward():
    rng = np.random.default_rng(1)
    tok = PlanTokenizer()
    batch = Collator(tok)([make_instance(rng, 5, 7)])
    model = tiny_model(tok).eval()
    with torch.no_grad():
        enc = model.encode(batch)
        full = model.decode(batch["input_ids"], enc)
        cache = model.new_cache()
        steps = [model.decode(batch["input_ids"][:, t : t + 1], enc, cache=cache, offset=t) for t in range(batch["input_ids"].shape[1])]
    inc = torch.cat(steps, dim=1)
    assert torch.allclose(full, inc, atol=1e-4)


def test_generate_produces_parseable_plans():
    rng = np.random.default_rng(2)
    tok = PlanTokenizer()
    instances = [make_instance(rng, 4, 6), make_instance(rng, 7, 3)]
    batch = Collator(tok)(instances)
    model = tiny_model(tok).eval()
    plans = generate(model, tok, batch, max_len=300)
    for plan, inst in zip(plans, instances):
        P = plan_to_dense(plan, inst["n"], inst["m"])
        assert P.shape == (inst["n"], inst["m"])
        assert (P >= 0).all()
        metrics = evaluate_instance(P, inst)
        assert "cost_gap" in metrics and "max_violation" in metrics


def test_transpose_symmetry_of_data():
    rng = np.random.default_rng(3)
    inst = make_instance(rng, 6, 4)
    t = transpose_instance(inst)
    P = plan_to_dense((inst["plan_rows"], inst["plan_cols"], inst["plan_vals"]), 6, 4)
    Pt = plan_to_dense((t["plan_rows"], t["plan_cols"], t["plan_vals"]), 4, 6)
    assert np.allclose(P.T, Pt)
    assert np.allclose(P.sum(1), inst["a"]) and np.allclose(Pt.sum(1), t["a"])


def test_symmetric_prediction_runs():
    rng = np.random.default_rng(4)
    tok = PlanTokenizer()
    model = tiny_model(tok).eval()
    plans = predict_plans(model, tok, [make_instance(rng, 5, 5)], torch.device("cpu"), max_len=200, symmetric=True)
    assert plans[0].shape == (5, 5)
