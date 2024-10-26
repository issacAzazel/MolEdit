# Basic code for filter.

import jax
import jax.numpy as jnp
import numpy as np

from typing import Union, Callable
from jax.numpy import ndarray
from jax.nn.initializers import xavier_uniform, zeros
from flax import linen as nn

_FILTER_BY_KEY = dict()

def _filter_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _FILTER_BY_KEY:
            _FILTER_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _FILTER_BY_KEY:
                _FILTER_BY_KEY[alias] = cls
        return cls
    return alias_reg

class Filter(nn.Module):
    r"""Base class for filter network.

    ## Args:
        dim_in (int):    Number of basis functions.

        dim_out (int):   Dimension of output filter Tensor.

        activation (hk.Module or function):  Activation function. Default: None
        
        name (str):      Name of the filter network. Default: "filter".

    """

    dim_in: int
    dim_out: int
    activation: Union[Callable, str]
    
    def __call__(self, x: ndarray):
        r"""Compute filter.
        ## Args:
            x (ndarray):    Array of shape (...,). Input array.
        """
        raise NotImplementedError



