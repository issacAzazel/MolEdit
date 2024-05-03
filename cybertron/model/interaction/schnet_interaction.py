# Basic code for schnet interaction

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from typing import Optional, Union, Tuple, Callable
from ...common.filter.dense import DenseFilter
from ...common.layers.mlp import MLP

class SchnetInteraction(nn.Module):
    r"""Interaction layer of Schnet.

    ## Args:
        dim_filter (int): The dimension of the filter.
        
        activation (Callable, str): The activation function used.

        normalize_filter (bool): Whether to normalize the filter.

    """

    dim_filter: int
    activation: Union[Callable, str] = 'silu'
    normalize_filter: bool = False

    @nn.compact
    def __call__(self, node_vec, node_mask, edge_vec, edge_mask, edge_cutoff):
        r"""
        
        """
        dim_edge_rep = edge_vec.shape[-1]
        dim_node_rep = node_vec.shape[-1]
        dim_filter = self.dim_filter if self.dim_filter is not None else dim_edge_rep

        filter_net = DenseFilter(dim_in=dim_edge_rep, dim_out=dim_filter, activation=self.activation)
        atomwise_bc = nn.Dense(features=dim_filter)
        atomwise_ac = MLP(output_sizes=(dim_node_rep, dim_node_rep), activation=self.activation, activate_final=False)
        
        def _aggregate(inputs, mask):

            if mask is not None:
                # (A, A, W) * (A, A, 1)
                inputs = inputs * jnp.expand_dims(mask, -1)
            _out = jnp.sum(inputs, axis=-2)

            if self.normalize_filter:
                if mask is not None:
                    num = jnp.sum(mask, axis=-2)
                    num = jnp.maximum(num, 1) ## Liyh: need to check this
                else:
                    num = inputs.shape[-2]
                _out = _out / num
            
            return _out
            
        # (A, F) -> (A, W)
        x_i = atomwise_bc(node_vec)
        # (A, A, K) -> (A, A, W)
        g_ij = filter_net(edge_vec)
        # (A, A, W) * (A, A, 1) -> (A, A, W)
        w_ij = g_ij * jnp.expand_dims(edge_cutoff, -1)
        # (1, A, W) * (A, A, W)
        y = jnp.expand_dims(x_i, -3) * w_ij
        # (A, A, W) -> (A, W)
        y = _aggregate(y, edge_mask)
        # (A, W) -> (A, F)
        y = atomwise_ac(y)
        # (A, F) + (A, F) -> (A, F)
        node_new = node_vec + y

        return node_new, edge_vec