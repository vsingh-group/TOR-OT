"""Synthetic OT instance generation and data loading.

Only the torch-free parts are imported eagerly so that ``torot.data.generate`` can run on
a CPU machine without a PyTorch installation; import ``torot.data.dataset`` explicitly
for the PyTorch loaders.
"""

from torot.data.cost import COST_TYPES, cost_matrix

__all__ = ["COST_TYPES", "cost_matrix"]
