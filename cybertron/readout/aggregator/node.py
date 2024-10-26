# Code for node aggregators.

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from typing import Optional, Union, Tuple, List
from ...common.activation import get_activation

__all__ = [
    'NodeSummation',
    'NodeMean',
]

_AGGREGATOR_BY_KEY = dict()

def _aggregator_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _AGGREGATOR_BY_KEY:
            _AGGREGATOR_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _AGGREGATOR_BY_KEY:
                _AGGREGATOR_BY_KEY[alias] = cls

        return cls

    return alias_reg

@_aggregator_register('node_sum')
class NodeSummation(nn.Module):
    r"""Node summation aggregator.
    
    ## Args:

        reduce_axis (int): Axis to reduce. Default: -2.

        name (str): Name of the module. Default: 'node_sum_aggregator'.

    """

    reduce_axis: int = -2
    name: str = 'node_sum_aggregator'
    
    def __call__(self,
                 node_vec: jnp.ndarray,
                 node_mask: Optional[jnp.ndarray] = None,
                 num_atoms: Optional[jnp.ndarray] = None,
                 ):
        r"""
        ## Args:

            node_vec: Node vectors. Shape: (A, F).

            node_mask: Node masks. Shape: (A,).

            num_atoms: Number of atoms. Shape: (,).

        """

        if node_mask is not None:
            # (A, F) * (A, 1) -> (A, F)
            node_vec = node_vec * jnp.expand_dims(node_mask, axis=-1)
        
        # (A, F) -> (F,)
        agg = jnp.sum(node_vec, axis=self.reduce_axis)

        return agg
    
    def __str__(self) -> str:
        return "NodeSummation<>"

@_aggregator_register('node_mean')
class NodeMean(nn.Module):
    r"""Node mean aggregator.

    ## Args:

        reduce_axis (int): Axis to reduce. Default: -2.

        name (str): Name of the module. Default: 'node_mean_aggregator'.
        
    """

    reduce_axis: int = -2
    name: str = 'node_mean_aggregator'
    
    def __call__(self,
                 node_vec: jnp.ndarray,
                 node_mask: Optional[jnp.ndarray] = None,
                 num_atoms: Optional[jnp.ndarray] = None,
                 ):
        """
        ## Args:

            node_vec: Node vectors. Shape: (A, F).

            node_mask: Node masks. Shape: (A,).

            num_atoms: Number of atoms. Shape: (,).   
        """

        if node_mask is not None:
            # (A, F) * (A, 1) -> (A, F)
            node_vec = node_vec * jnp.expand_dims(node_mask, axis=-1)
        
        # (A, F) -> (F,)
        agg = jnp.sum(node_vec, axis=self.reduce_axis) / num_atoms

        return agg
    
    def __str__(self) -> str:
        return "NodeMean<>"