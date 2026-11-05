# Vendored: UNOT, Universal Neural Optimal Transport (official code)

* Repository: https://github.com/GregorKornhardt/UNOT
* Commit: 73b290ba8c0709fa7dd4992eeda88171628d6bb2
* License: MIT (`LICENSE`). Copyright (c) 2025 Gregor Kornhardt.
* Paper: J. Geuter, G. Kornhardt, I. Tomasson, V. Laschos. *Universal Neural Optimal Transport.* ICML 2025.

Files copied verbatim (no modification): `src/loss.py`, `src/networks/{generator,mlp}.py`, `src/train/train_mlp.py`,
`src/ot/{sinkhorn,cost_matrix}.py`, `scripts/main_mlp.py` (reference for the default hyperparameters), `README.md`,
`LICENSE`. The adapter `baselines/unot/train.py` imports them unchanged; everything specific to ToR-OT's setting
(1D histograms, cost matrices, 1D generator skip connection, evaluation) lives in the adapter.
