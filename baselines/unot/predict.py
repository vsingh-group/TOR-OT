"""UNOT predictions on the evaluation sets (run in env_unot, from the repository root).

    python -m baselines.unot.predict --root experiments/torot/camera_ready \\
        --runs 'experiments/baselines/unot/*' --sets 'id_*' 'ood_paper_*'

For each set, the run trained on the set's cost with the lowest validation MAPE is used (recorded in the timing
file). Plans: v = exp(predictor(mu, nu)), u = mu / (K v), P = diag(u) K diag(v) (see train.plans_from_predictor).
Writes <root>/preds/<set>__unot.npy and <set>__unot.timing.json. With UNOT_SINKHORN_STEPS=k (environment), the plan after
k further Sinkhorn iterations from UNOT's prediction (its intended use), as <set>__unot_sk<k>.
"""
import os

import torch

from baselines.common import cost_matrix, predict_cli, predict_sets
from baselines.unot.train import load, plans_from_predictor

if __name__ == "__main__":
    args = predict_cli(__doc__.split("\n")[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_run(run):
        predictor, meta = load(run, device)
        return predictor, cost_matrix(meta["n"], meta["cost"], meta.get("dim", 1)), meta["eps"]

    def predict(model, a, b):
        predictor, C, eps = model
        plans = plans_from_predictor(predictor, a, b, C, eps, device, sinkhorn_steps=STEPS)
        torch.cuda.synchronize() if device.type == "cuda" else None
        return plans

    STEPS = int(os.environ.get("UNOT_SINKHORN_STEPS", "0"))      # > 0: UNOT + that many Sinkhorn iterations
    predict_sets(args.method or ("unot" if STEPS == 0 else f"unot_sk{STEPS}"), args.root, args.runs, args.sets, load=load_run,
                 predict=predict, device=torch.cuda.get_device_name() if device.type == "cuda" else "cpu", note=args.note)
