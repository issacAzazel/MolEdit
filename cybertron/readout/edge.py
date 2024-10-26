# Code for edge readout.

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from typing import Optional, Union, Callable, Tuple
from ..common.activation import get_activation
from .readout import Readout, _readout_register
from .aggregator import get_aggregator
from .decoder import Decoder, get_decoder

@_readout_register('pairwise')
class PairwiseReadout(Readout):
    r"""Pairwise readout module.

    ## Args:
        
            dim_node_rep (int): Dimension of node vectors.
    
            dim_edge_rep (int): Dimension of edge vectors.

            dim_output (int): Dimension of outputs. Default: 1

            activation (Callable, str): Activation function, Default: silu
    
            decoder (str): Decoder method. Default: 'halve'
    
            aggregator (str): Aggregator method. Default: 'edge_sum'

            reduce_axis (tuple): Axis to reduce. Default: (-3, -2)

            name (str): Name of the module. Default: 'pairwise_readout'
    """

    dim_node_rep: int
    dim_edge_rep: int
    dim_output: int = 1
    activation: Union[Callable, str] = 'silu'
    decoder: Union[Decoder, dict, str] = 'halve'
    aggregator: Union[nn.Module, dict, str] = 'edge_sum'
    reduce_axis: tuple = (-3, -2)
    name: str = 'pairwise_readout'

    def setup(self):
        
        self.decoder_net = get_decoder(cls_name=self.decoder,
                                       dim_in=self.dim_edge_rep,
                                       dim_out=self.dim_output,
                                       activation=self.activation,
                                       name="decoder",)
        self.aggregator_fn = get_aggregator(cls_name=self.aggregator,
                                            axis=self.reduce_axis,
                                            name="aggregator",)
        
    def __call__(self,
                 node_vec: jax.Array,
                 edge_vec: jax.Array,
                 node_mask: jax.Array,
                 edge_mask: jax.Array,
                 edge_cutoff: jax.Array,
                 ):
        r"""
        ## Args:
        
            edge_vec (jax.Array): Edge vectors with shape (A, A, F)
            
            edge_mask (jax.Array): Edge mask with shape (A, A)
            
            edge_cutoff (jax.Array): Edge cutoff with shape (A, A)
        """
        # (A, A) -> (,)
        if edge_mask is None:
            _A = edge_vec.shape[-2]
            num_edges = _A * _A
        else:
            num_edges = jnp.sum(edge_mask.astype(jnp.int32), axis=(-2, -1))

        y = edge_vec
        if self.decoder_net is not None:
            # (A, A, F) -> (A, A, Y)
            y = self.decoder_net(y)
        
        if self.aggregator_fn is not None:
            # (A, A, Y) -> (Y,)
            y = self.aggregator_fn(y, edge_mask, num_edges, edge_cutoff)
        
        return y