# coding=utf-8
"""Common utils for GPtorch."""
from typing import NamedTuple, Optional, Tuple, Union, Any, Sequence, Dict, List
import torch
import numpy as np

Array = torch.Tensor

class SubDataset(NamedTuple):
  """Sub dataset with x: n x d and y: n x m; d, m>=1."""
  x: torch.Tensor
  y: torch.Tensor
  aligned: Optional[Union[int, str, Tuple[str, ...]]] = None

Dataset = Sequence[SubDataset]

def _tree_flatten(tree):
    """Flattens a nested dictionary/list structure."""
    leaves = []
    spec = []
    if isinstance(tree, dict):
        keys = sorted(tree.keys())
        spec.append(('dict', keys))
        for k in keys:
            l, s = _tree_flatten(tree[k])
            leaves.extend(l)
            spec.extend(s)
    elif isinstance(tree, (list, tuple)):
        spec.append(('list', len(tree)))
        for v in tree:
            l, s = _tree_flatten(v)
            leaves.extend(l)
            spec.extend(s)
    else:
        leaves.append(tree)
        spec.append(('leaf', None))
    return leaves, spec

def _tree_unflatten(spec, leaves):
    """Unflattens a list of leaves into the structure defined by spec."""
    # This is a simplified unflatten that consumes leaves and spec recursively
    # We need a mutable iterator for leaves and spec to consume them
    leaves_iter = iter(leaves)
    spec_iter = iter(spec)
    
    def _recurse():
        try:
            type_, meta = next(spec_iter)
        except StopIteration:
            return None

        if type_ == 'leaf':
            return next(leaves_iter)
        elif type_ == 'dict':
            keys = meta
            d = {}
            for k in keys:
                # Recursively reconstruct the value for each key
                # Note: The spec structure is flattened depth-first in _tree_flatten
                # So we just call _recurse for each child
                d[k] = _recurse()
            return d
        elif type_ == 'list':
            length = meta
            l = []
            for _ in range(length):
                l.append(_recurse())
            return l
        else:
            raise ValueError(f"Unknown type {type_}")

    return _recurse()


class ParamsTree:
  """Converts between dictionary and flat array.
  
  Mirroring GPax ParamsTree functionality for PyTorch.
  """

  def __init__(self, params: Dict[str, Any]):
    self.leaves, self.spec = _tree_flatten(params)
    # Store shapes of tensors
    self.shapes = [p.shape for p in self.leaves]
    # Calculate sizes
    sizes = np.array([np.prod(s) for s in self.shapes])
    self.indices = np.cumsum(sizes)[:-1]

  def toarray(self, params: Dict[str, Any]) -> torch.Tensor:
    leaves, _ = _tree_flatten(params)
    # Flatten each tensor and concatenate
    flat_params = [p.reshape(-1) for p in leaves]
    if not flat_params:
        return torch.tensor([])
    return torch.cat(flat_params)

  def todict(self, params: torch.Tensor) -> Dict[str, Any]:
    # Split the flat array back into chunks
    if params.numel() == 0:
        leaves = []
    else:
        # torch.split expects sizes, not indices (unless using tensor_split equivalent)
        # But here we calculated split indices. Let's convert back to sizes.
        # Or just use tensor_split with indices if using CPU/numpy?
        # torch.tensor_split is available in recent pytorch.
        
        # Let's use split with sizes, it's safer.
        sizes = [np.prod(s) for s in self.shapes]
        chunks = torch.split(params, sizes)
        
        leaves = [c.reshape(s) for c, s in zip(chunks, self.shapes)]
        
    return _tree_unflatten(self.spec, leaves)

  def __repr__(self) -> str:
    return f'shapes={self.shapes}, indices={self.indices}'


def constant_initializer_factory(constant: float):
  """Returns a function that initializes a tensor with a constant value."""
  def initializer(shape, dtype=torch.float32):
    return torch.full(shape, constant, dtype=dtype)
  return initializer

