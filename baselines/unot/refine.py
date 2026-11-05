"""UNOT followed by k Sinkhorn iterations (run in env_unot, from the repository root).

    python -m baselines.unot.refine --root experiments/torot/camera_ready --runs 'experiments/baselines/unot/*' \\
        --sets 'id_simplex_*' 'ood_paper_*'

UNOT's prediction initialises Sinkhorn: v = exp(predictor(mu, nu)), then the official iterations u = mu / (K v),
v = nu / (K^T u) (src/ot/sinkhorn.py), computed here in the log domain in float64 (identical iterates, no underflow
for small eps). k = 0 is the plan of predict.py. Also runs Sinkhorn from its standard start (v = 1) at the same eps.
Writes <root>/warmstart/<set>__unot.json and <set>__sinkhorn_cold_eps<eps>.json.
"""
import torch

from baselines.common import cost_matrix, predict_cli, refine_sets
from baselines.unot.train import load

if __name__ == "__main__":
    args = predict_cli(__doc__.split("\n")[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_run(run):
        predictor, meta = load(run, device)
        logK = -torch.as_tensor(cost_matrix(meta["n"], meta["cost"], meta.get("dim", 1)), dtype=torch.float64, device=device) / meta["eps"]
        return predictor, logK

    def t(x):
        return torch.as_tensor(x, dtype=torch.float64, device=device)

    @torch.no_grad()
    def init(model, a, b, warm):
        predictor, logK = model
        A = t(a)
        log_v = predictor(A.float(), t(b).float()).double() if warm else torch.zeros_like(A)
        log_u = torch.log(A) - torch.logsumexp(logK[None] + log_v[:, None, :], dim=2)
        return log_u, log_v

    @torch.no_grad()
    def iterate(model, state, a, b, k):
        _, logK = model
        log_u, log_v = state
        la, lb = torch.log(t(a)), torch.log(t(b))
        for _ in range(k):
            log_v = lb - torch.logsumexp(logK[None] + log_u[:, :, None], dim=1)
            log_u = la - torch.logsumexp(logK[None] + log_v[:, None, :], dim=2)
        return log_u, log_v

    def plan(model, state):
        _, logK = model
        log_u, log_v = state
        return torch.exp(log_u[:, :, None] + logK[None] + log_v[:, None, :])

    def sync(x):
        torch.cuda.synchronize() if device.type == "cuda" else None
        return x.cpu().numpy()

    refine_sets("unot", args.root, args.runs, args.sets, load=load_run, init=init, iterate=iterate,
                plan=plan, sync=sync,
                device=torch.cuda.get_device_name() if device.type == "cuda" else "cpu")
