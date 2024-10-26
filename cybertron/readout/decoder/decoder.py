# Basic code for decoders.

import jax
import jax.numpy as jnp
import jax.nn as nn
import numpy as np

from flax import linen as nn
from jax.numpy import ndarray
from typing import Optional, Union, Tuple, List, Callable
from ...common.activation import get_activation

_DECODER_BY_KEY = dict()

def _decoder_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _DECODER_BY_KEY:
            _DECODER_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _DECODER_BY_KEY:
                _DECODER_BY_KEY[alias] = cls
        return cls
    return alias_reg

class Decoder(nn.Module):
    r"""Decoder network to reduce the representation vector.

    ## Args:

        dim_in (int): Input dimension.

        dim_out (int): Output dimension.

        activation (callable or str): Activation function.

        n_layers (int): Number of hidden layers. Default: 1.

        name (str): Name of the module.
    """

    dim_in: int
    dim_out: int = 1
    n_layers: int = 1
    name: str = "decoder"
    
    @nn.compact
    def __call__(self, x: ndarray):
        
        raise NotImplementedError
    
    def __str__(self) -> str:
        return 'Decoder<>'