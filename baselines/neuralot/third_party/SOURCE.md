# Vendored: Neural Optimal Transport (official code)

* Repository: https://github.com/iamalexkorotin/NeuralOptimalTransport
* License: MIT (`LICENSE`, copied as published: "Copyright (c) 2023 Alexander").
* Paper: A. Korotin, D. Selikhanovych, E. Burnaev. *Neural Optimal Transport.* ICLR 2023.

`src/`: the repository's `src/` folder. Seven of its eight files are identical to upstream commit
`13850338a577b607a9a409c4c6cdfa26c60fc226`; `plotters.py` adds the 1D plotting helpers used by the scripts below.

`NOT_toy_1D_cost_strong.py` and `NOT_toy_2D_cost_strong.py`: our scripts, adapted from the repository's
`notebooks/NOT_toy_1D.ipynb` and `notebooks/NOT_toy_2D.ipynb` (strong cost, histograms on a grid); they are called by
[`predict.py`](../predict.py). Empty `__init__.py` files were added so the folder imports as a package.
