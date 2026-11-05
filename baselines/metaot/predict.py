"""Meta OT predictions on the evaluation sets (run in env_metaot, from the repository root).

    python -m baselines.metaot.predict --root experiments/torot/camera_ready \\
        --runs 'experiments/baselines/metaot/*' --sets 'id_*' 'ood_paper_*'

For each set, the run trained on the set's cost with the lowest validation MAPE is used (recorded in the timing
file). Plans are the official pred_transport: exp((f + g(f) - C) / eps), f from the network, g its c-transform.
Writes <root>/preds/<set>__metaot.npy and <set>__metaot.timing.json.
"""
import jax

from baselines.common import predict_cli, predict_sets
from baselines.metaot.train import Workspace1D

if __name__ == "__main__":
    args = predict_cli(__doc__.split("\n")[0])
    predict_sets(args.method or "metaot", args.root, args.runs, args.sets, load=lambda run: Workspace1D.load(run)[0],
                 predict=lambda ws, a, b: ws.predict(a, b), device=str(jax.devices()[0]), note=args.note)
