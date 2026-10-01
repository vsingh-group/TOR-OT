"""Print the training progress of a run directory from its ``log.jsonl``.

    python -m torot.progress experiments/torot/base_25ep            # summary + validation table
    python -m torot.progress experiments/torot/base_25ep --last 5   # only the last 5 validations
    watch -n 60 python -m torot.progress experiments/torot/base_25ep

For a one-glance status, ``cat <run_dir>/progress.txt`` (written by the trainer).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from omegaconf import OmegaConf


def load_log(run_dir: str) -> tuple[List[dict], List[dict]]:
    train, val = [], []
    with open(os.path.join(run_dir, "log.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            (val if "val/loss" in rec else train).append(rec)
    return train, val


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--last", type=int, default=None, help="show only the last N validations")
    args = p.parse_args(argv)

    progress = os.path.join(args.run_dir, "progress.txt")
    if os.path.exists(progress):
        print(open(progress).read())

    try:
        train, val = load_log(args.run_dir)
    except FileNotFoundError:
        sys.exit(f"no log.jsonl in {args.run_dir}")
    max_steps = OmegaConf.load(os.path.join(args.run_dir, "config.yaml")).optim.max_steps

    if train:
        t = train[-1]
        print(f"train: step {t['step']}/{max_steps}  loss {t['loss']:.4f}  grad_norm {t['grad_norm']:.2f}  "
              f"acc ptr {t.get('acc/tgt_ptr', float('nan')):.3f} exp {t.get('acc/exponent', float('nan')):.3f} "
              f"d1 {t.get('acc/digit1', float('nan')):.3f}  elapsed {t['elapsed_s'] / 3600:.2f} h")
    if not val:
        print("no validation yet")
        return

    rows = val if args.last is None else val[-args.last :]
    best = min(val, key=lambda r: r["val/loss"])
    header = f"{'step':>7} {'val_loss':>9} {'ptr_acc':>8} {'exp_acc':>8} {'d1_acc':>7} {'mape_med':>9} {'gap_med':>10} {'viol_med':>9} {'recall':>7}"
    print("\nvalidations" + (f" (last {args.last})" if args.last else "") + f"  |  best val loss {best['val/loss']:.4f} at step {best['step']}")
    print(header)
    for r in rows:
        print(f"{r['step']:>7} {r['val/loss']:>9.4f} {r.get('val/acc/tgt_ptr', 0):>8.3f} {r.get('val/acc/exponent', 0):>8.3f} "
              f"{r.get('val/acc/digit1', 0):>7.3f} {r.get('val/mape_ref_support/median', float('nan')):>9.4f} "
              f"{r.get('val/cost_gap/median', float('nan')):>+10.2e} {r.get('val/max_violation/median', float('nan')):>9.2e} "
              f"{r.get('val/support_recall/mean', float('nan')):>7.3f}"
              + ("  *" if r is best else ""))


if __name__ == "__main__":
    main()
