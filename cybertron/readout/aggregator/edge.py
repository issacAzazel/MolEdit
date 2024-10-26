# Code for edge aggregators.

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from typing import Optional, Union, Tuple, List
from ...common.activation import get_activation
from .node import _aggregator_register

__all__ = [
    'EdgeSummation',
    'EdgeMean',
]

@_aggregator_register('edge_sum')
class EdgeSummation(nn.Module):

    reduce_axis: tuple = (-3, -2)
    name: str = 'edge_sum_aggregator'

    @nn.compact
    def __call__(self,
                 edge_vec: jax.Array,
                 edge_mask: Optional[jax.Array] = None,
                 num_edges: Optional[jax.Array] = None,
                 edge_cutoff: Optional[jax.Array] = None,
                 ):
        r"""
        ## Args:

            edge_vec (jax.Array): Edge vectors. Shape: (A, A, F).

            edge_mask (jax.Array): Edge mask. Shape: (A, A).

            edge_cutoff (jax.Array): Edge cutoff. Shape: (A, A).
        """

        if edge_cutoff is not None:
            # (A, A) * (A, A) -> (A, A) -> (A, A, 1)
            edge_cutoff = edge_cutoff * edge_mask if edge_mask is not None else edge_cutoff
            edge_cutoff = jnp.expand_dims(edge_cutoff, axis=-1)
            # (A, A, 1) * (A, A, F) -> (A, A, F)
            edge_vec = edge_vec * edge_cutoff
        elif edge_mask is not None:
            edge_vec = edge_vec * edge_mask
        else:
            edge_vec = edge_vec
        
        # (A, A, F) -> (F,)
        agg = jnp.sum(edge_vec, axis=self.reduce_axis)
        return agg


@_aggregator_register('edge_mean')
class EdgeMean(nn.Module):

    reduce_axis: tuple = (-3, -2)
    name: str = 'edge_mean_aggregator'

    @nn.compact
    def __call__(self,
                 edge_vec: jax.Array,
                 edge_mask: Optional[jax.Array] = None,
                 num_edges: Optional[jax.Array] = None,
                 edge_cutoff: Optional[jax.Array] = None,
                 ):
        r"""
        ## Args:

            edge_vec (jax.Array): Edge vectors. Shape: (A, A, F).

            edge_mask (jax.Array): Edge mask. Shape: (A, A).

            edge_cutoff (jax.Array): Edge cutoff. Shape: (A, A).
        """
        if edge_cutoff is not None:
            # (A, A) * (A, A) -> (A, A)
            edge_cutoff = edge_cutoff * edge_mask if edge_mask is not None else edge_cutoff
            edge_cutoff = jnp.expand_dims(edge_cutoff, axis=-1)
            # (A, A) * (A, A, F) -> (A, A, F)
            edge_vec = edge_vec * edge_cutoff
        elif edge_mask is not None:
            edge_vec = edge_vec * edge_mask
            if num_edges is None:
                num_edges = jnp.sum(edge_mask, axis=self.reduce_axis)
        else:
            edge_vec = edge_vec
        
        if num_edges is None:
            raise ValueError("[readout/aggregator/edge] num_edges must be provided if edge_mask is not provided.")
        
        # (A, A, F) -> (F,)
        agg = jnp.sum(edge_vec, axis=self.reduce_axis)
        agg = agg / num_edges

        return agg
