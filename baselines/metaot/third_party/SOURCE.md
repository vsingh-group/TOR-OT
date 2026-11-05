# Vendored: Meta Optimal Transport (official code)

* Repository: https://github.com/facebookresearch/meta-ot
* Commit: 7dcc9a4c25b0a57626d3fa26028c6633ad19da50
* License: CC BY-NC 4.0 (`LICENSE`), non-commercial use only. Copyright (c) Meta Platforms, Inc. and affiliates.
* Paper: B. Amos, S. Cohen, G. Luise, I. Redko. *Meta Optimal Transport.* ICML 2023.

Files copied verbatim (no modification): `train_discrete.py`, `conf/train_discrete.yaml`, `meta_ot/*.py`,
`README.md`, `LICENSE`. The adapter `baselines/metaot/train.py` imports them unchanged; everything specific to
ToR-OT's setting (1D histograms, cost matrices, sampler, evaluation) lives in the adapter.
