# Basic code for node readout.

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from typing import Optional, Union, Callable
from ..common.activation import get_activation
from .readout import Readout, _readout_register
from .aggregator import get_aggregator
from .decoder import Decoder, get_decoder

@_readout_register('atomwise')
class AtomwiseReadout(Readout):
    r"""Atomwise readout module.

    ## Args:

        dim_output (int): Dimension of outputs. Default: 1

        dim_node_rep (int): Dimension of node vectors. Default: None

        activation (Callable, str): Activation function, Default: None

        decoder (str): Decoder method. Default: 'halve'

        aggregator (str): Aggregator method. Default: 'sum'
    
    """

    dim_node_rep: int
    dim_output: int = 1
    num_layers: int = 1
    activation: Union[Callable, str] = 'silu'
    decoder: Union[Decoder, dict, str] = 'halve'
    aggregator: Union[nn.Module, dict, str] = 'node_sum'
    name: str = 'atomwise_readout'

    def setup(self):
        
        self.decoder_net = get_decoder(cls_name=self.decoder,
                                       dim_in=self.dim_node_rep,
                                       dim_out=self.dim_output,
                                       activation=self.activation,
                                       n_layers=self.num_layers,
                                       name="decoder",)
        
        self.aggregator_fn = get_aggregator(cls_name=self.aggregator,
                                            axis=-2,
                                            name="aggregator",)

    def __call__(self,
                 node_vec: jax.Array,
                 edge_vec: jax.Array,
                 node_mask: jax.Array,
                 edge_mask: jax.Array,
                 edge_cutoff: jax.Array,):
        r"""
        
        ## Args:
        
            node_vec (jax.Array):    Node vectors. Shape: (A, F)

            node_mask (jax.Array):   Node masks. Shape: (A,)
        """
        
        if node_mask is None:
            num_atoms = node_vec.shape[-2] # A
        else:
            num_atoms = jnp.sum(node_mask.astype(jnp.int32))
        
        y = node_vec
        if self.decoder_net is not None:
            y = self.decoder_net(y)

        if self.aggregator_fn is not None:
            y = self.aggregator_fn(y, node_mask, num_atoms)
        
        return y
    
    def __str__(self) -> str:
        return "AtomwiseReadout<>"
    
@_readout_register('graph')
class GraphReadout(Readout):
    r"""Atomwise readout module.

    ## Args:

        dim_output (int): Dimension of outputs. Default: 1

        dim_node_rep (int): Dimension of node vectors. Default: None

        activation (Callable, str): Activation function, Default: None

        decoder (str): Decoder method. Default: 'halve'

        aggregator (str): Aggregator method. Default: 'sum'
    
    """

    dim_node_rep: int
    dim_edge_rep: Optional[int] = None
    dim_output: int = 1
    num_layers: int = 1
    activation: Union[Callable, str] = 'silu'
    decoder: Union[Decoder, dict, str] = 'halve'
    aggregator: Union[nn.Module, dict, str] = 'node_mean'

    def setup(self):
        
        self.decoder_net = get_decoder(cls_name=self.decoder,
                                      dim_in=self.dim_node_rep,
                                      dim_out=self.dim_output,
                                      activation=self.activation,
                                      n_layers=self.num_layers,
                                      name="decoder",)
        
        self.aggregator_fn = get_aggregator(cls_name=self.aggregator,
                                            axis=-2,
                                            name="aggregator",)

    def __call__(self,
                 node_vec: jax.Array,
                 edge_vec: jax.Array,
                 node_mask: jax.Array,
                 edge_mask: jax.Array,
                 edge_cutoff: jax.Array,):
        r"""
        
        ## Args:
        
            node_vec (jax.Array):    Node vectors. Shape: (A, F)

            node_mask (jax.Array):   Node masks. Shape: (A,)
        """
        
        if node_mask is None:
            num_atoms = node_vec.shape[-2] # A
        else:
            num_atoms = jnp.sum(node_mask.astype(jnp.int32))
        
        y = node_vec
        if self.aggregator_fn is not None:
            y = self.aggregator_fn(y, node_mask, num_atoms)
        else:
            raise ValueError("[GraphReadout] Aggregator is None!")
        
        if self.decoder_net is not None:
            y = self.decoder_net(y)
        
        return y
    
    def __str__(self) -> str:
        return "GraphReadout<>"

        