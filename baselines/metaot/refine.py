"""Meta OT followed by k Sinkhorn iterations (run in env_metaot, from the repository root).

    python -m baselines.metaot.refine --root experiments/torot/camera_ready --runs 'experiments/baselines/metaot/*' \\
        --sets 'id_simplex_*' 'ood_paper_*'

Meta OT's prediction is meant to initialise Sinkhorn: f from the network, then alternating potential updates with the
official geometry (ott-jax ``update_potential``, log domain), exactly the iterations of train_discrete.py's
``pred_error`` / ``g_from_f``. k = 0 is the plan of predict.py. Also runs Sinkhorn from its standard start (f = 0) at
the same eps. Writes <root>/warmstart/<set>__metaot.json and <set>__sinkhorn_cold_eps<eps>.json.
"""
import jax
import jax.numpy as jnp

from baselines.common import predict_cli, refine_sets
from baselines.metaot.train import Workspace1D


def make_fns():
    cache = {}

    def fns(ws):
        if id(ws) not in cache:
            geom = ws.geom

            def init_one(a, b, warm):
                f = ws.potential_model.apply({"params": ws.params}, a, b) if warm else jnp.zeros_like(a)
                return f, ws.g_from_f(a, b, f)

            def iterate_one(f, g, a, b, k):
                def body(_, fg):
                    f, g = fg
                    f = geom.update_potential(f, g, jnp.log(a), 0, axis=1)
                    g = geom.update_potential(f, g, jnp.log(b), 0, axis=0)
                    return f, g
                return jax.lax.fori_loop(0, k, body, (f, g))

            cache[id(ws)] = {
                True: jax.jit(jax.vmap(lambda a, b: init_one(a, b, True))),
                False: jax.jit(jax.vmap(lambda a, b: init_one(a, b, False))),
                "iterate": jax.jit(jax.vmap(iterate_one, in_axes=(0, 0, 0, 0, None)), static_argnums=4),
                "plan": jax.jit(jax.vmap(geom.transport_from_potentials)),
            }
        return cache[id(ws)]
    return fns


if __name__ == "__main__":
    args = predict_cli(__doc__.split("\n")[0])
    fns = make_fns()
    refine_sets(
        "metaot", args.root, args.runs, args.sets, load=lambda run: Workspace1D.load(run)[0],
        init=lambda ws, a, b, warm: fns(ws)[warm](jnp.asarray(a), jnp.asarray(b)),
        iterate=lambda ws, st, a, b, k: st if k == 0 else fns(ws)["iterate"](st[0], st[1], jnp.asarray(a), jnp.asarray(b), k),
        plan=lambda ws, st: fns(ws)["plan"](*st),
        sync=lambda x: jax.block_until_ready(x).__array__(),
        device=str(jax.devices()[0]))
