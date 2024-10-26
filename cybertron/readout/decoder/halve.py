# Code for halve decoder.

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from typing import Optional, Union, Tuple, List, Callable
from ...common.activation import get_activation
from .decoder import Decoder, _decoder_register
from ...common.layers.mlp import MLP

@_decoder_register("halve")
class HalveDecoder(Decoder):
    r"""A MLP decoder with halve number of layers.

    ## Args:

        dim_in (int): Input dimension.

        dim_out (int): Output dimension. Default: 1.

        activation (Union[Callable, str]): Activation function. Default: None.

        n_layers (int): Number of hidden layers. Default: 1.

        name (str): Name of the module. Default: "halve_decoder".

    """

    dim_in: int
    dim_out: int = 1
    activation: Union[Callable, str] = nn.silu
    n_layers: int = 1
    name: str = "halve_decoder"

    def setup(self):

        if isinstance(self.activation, str):
            self.act_fn = get_activation(self.activation)
        elif isinstance(self.activation, Callable):
            self.act_fn = self.activation

        if self.n_layers > 0:
            n_hiddens = []
            dim = self.dim_in

            for _ in range(self.n_layers):
                dim = dim // 2
                if dim < self.dim_out:
                    raise ValueError("The dimension of hidden layer is smaller than output dimension.")
                n_hiddens.append(dim)
            n_hiddens.append(self.dim_out)
            
            self.output = MLP(output_sizes=n_hiddens,
                              activation=self.act_fn,
                              activate_final=False,
                              name="mlp",)
        else:
            self.output = nn.Sequential([
                nn.Dense(features=self.dim_out,
                         use_bias=True,
                         kernel_init=nn.initializers.xavier_uniform(),
                         name="linear",),
                self.act_fn,
            ])
        
    def __call__(self, x: jax.Array):

        return self.output(x)
    
    def __str__(self):
        return 'HalveDecoder<>'