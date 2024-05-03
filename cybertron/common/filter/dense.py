# Code for dense filter

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from typing import Optional, Type, Union, Callable
from jax.numpy import ndarray
from jax.nn.initializers import xavier_uniform, zeros
from .filter import Filter
from ..activation import get_activation
from ..layers.mlp import MLP

class DenseFilter(Filter):
    r"""Dense filter network.

    ## Args:
        dim_in (int):    Number of basis functions.

        dim_out (int):   Dimension of output filter Tensor.

        activation (hk.Module or function): Activation function. Default: None.
        
        n_hidden (int):  Number of hidden layers. Default: 1.
        
        name (str):      Name of the filter network. Default: "dense_filter".

    """

    dim_in: int
    dim_out: int
    activation: Union[Callable, str] = "relu"
    n_hidden: int = 1

    def setup(self):
        
        if self.n_hidden > 0:
            hidden_layers = [self.dim_out for _ in range(self.n_hidden + 1)]
            self.layers = MLP(output_sizes=hidden_layers,
                              activation=self.activation,
                              activate_final=False,
                              name="mlp",)
        else:
            self.layers = MLP(output_sizes=[self.dim_out],
                              activation=self.activation,
                              activate_final=True,
                              name="mlp",)
        
    def __call__(self, x: jax.Array) -> jax.Array:
        r"""Compute filter.
        ## Args:
            x (jax.Array):    Array of shape (...,). Input array.
        """
        return self.layers(x)