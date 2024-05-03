# Hyper Dense Module

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Union, Callable, Optional, Any
from flax.linen.initializers import lecun_normal, zeros_init
from ..activation import get_activation

Dtype = Any

class Dense(nn.Module):
    r"""A dense layer with activation function.
    
    ## Args:
        dim_out (int): Dimension of output vectors.

        activation (str, Callable): Activation function. Default: 'relu'.
    """

    dim_out: int
    use_bias: bool = True
    activation: Union[Callable, str] = "relu"
    d_type: Optional[Dtype] = jnp.float32
    kernel_init: Callable = lecun_normal()
    bias_init: Callable = zeros_init()

    @nn.compact
    def __call__(self, x):
        act_fn = get_activation(self.activation)
        linear_fn = nn.Dense(self.dim_out, self.use_bias, self.d_type, kernel_init=self.kernel_init, bias_init=self.bias_init)

        return act_fn(linear_fn(x)) 