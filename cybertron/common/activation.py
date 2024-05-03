# Code for get_activation and activation functions.

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from typing import Any, Optional, Union, Callable

## Shifted Softplus
class ShiftedSoftplus(nn.Module):

    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x):
        return nn.softplus(x + self.epsilon) - jnp.log(2.0)

@jax.jit
def ssp(x, epsilon: float = 1e-6):
    return nn.softplus(x + epsilon) - jnp.log(2.0)

_activation_dict = {
    'relu': nn.relu,
    'relu6': nn.relu6,
    'sigmoid': nn.sigmoid,
    'softplus': nn.softplus,
    'silu': nn.silu,
    'swish': nn.swish,
    'leaky_relu': nn.leaky_relu,
    'gelu': nn.gelu,
    'ssp': ssp,
}

def get_activation(name):
    """get activation function by name"""
    if name is None:
        raise ValueError("Activation name cannot be None!")
    if isinstance(name, str):
        if name.lower() in _activation_dict.keys():
            return _activation_dict[name.lower()]
        raise ValueError(
            "The activation corresponding to '{}' was not found.".format(name))
    if isinstance(name, Callable):
        return name
    raise TypeError("Unsupported activation type '{}'.".format(type(name)))

if __name__ == "__main__":
    print(get_activation('relu'))

    my_act = get_activation('relu')
    in_act = jnp.array([-1, 1, 2])
    print(my_act(in_act))