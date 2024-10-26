# Code for residual decoder.

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from jax.numpy import ndarray
from typing import Optional, Union, Tuple, List, Callable
from ...common.activation import get_activation
from .decoder import Decoder, _decoder_register

# utils for residual decoder
class PreActDense(nn.Module):

    dim: int
    idx: int = 1
    name: str = "preact_dense"
    activation: Union[Callable, str] = nn.silu

    def setup(self):

        if isinstance(self.activation, str):
            self.act_fn = get_activation(self.activation)
        elif isinstance(self.activation, Callable):
            self.act_fn = self.activation

        self.output = nn.Sequential([
                self.act_fn,
                nn.Dense(features=self.dim,
                         use_bias=True,
                         kernel_init=nn.initializers.xavier_uniform(),
                         name="linear_A",),
                self.act_fn,
                nn.Dense(features=self.dim,
                         use_bias=True,
                         kernel_init=nn.initializers.xavier_uniform(),
                         name="linear_B",),
            ])

    def __call__(self, x):

        return x + self.output(x)


@_decoder_register("residual")
class ResDecoder(Decoder):
    r"""A MLP decoder with residual connection.

    ## Args:

        dim_in (int): Input dimension.

        dim_out (int): Output dimension. Default: 1.

        activation (Union[Callable, str]): Activation function. Default: None.

        n_layers (int): Number of hidden layers. Default: 1.

        name (str): Name of the module. Default: "residual_decoder".

    """

    dim_in: int
    dim_out: int = 1
    activation: Union[Callable, str] = nn.silu
    n_layers: int = 1
    name: str = "residual_decoder"

    def setup(self):

        if isinstance(self.activation, str):
            self.act_fn = get_activation(self.activation)
        elif isinstance(self.activation, Callable):
            self.act_fn = self.activation
        
        self.layers = nn.Sequential([
            PreActDense(dim=self.dim_in, 
                        idx=i,
                        name=self.name, 
                        activation=self.act_fn)
            for i in range(self.n_layers)
        ])

        self.output = nn.Sequential([self.layers,
                                     self.act_fn,
                                     nn.Dense(features=self.dim_out,
                                              use_bias=True,
                                              kernel_init=nn.initializers.xavier_uniform(),
                                              name="last_linear",),
                                     ])
        
    def __call__(self, x: ndarray):
        r"""Compute decoder.

        Args:
            x (ndarray):    Array of shape (...,). Input array.
        """
        return self.output(x)
    
    def __str__(self) -> str:
        return 'ResDecoder<>'