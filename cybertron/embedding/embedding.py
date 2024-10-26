# Basic Code for embedding.

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from typing import Optional, Union, Callable
from ..model.interaction.molct_interaction import InteractionUnit
from ..common.rbf import get_rbf
from ..common.cutoff import get_cutoff
from ..common.filter import get_filter
from ..common.activation import get_activation

class Embedding(nn.Module):
    r"""Embedding Base.

    ## Args:

        fp_type:   Float type. Default: jnp.float32.
        
        int_type:   Int type. Default: jnp.int32.

        dim_node (int):   Node embedding dimension.

        dim_edge (int):   Edge embedding dimension.

        activation (str, Callable):  Activation function. Default: 'silu'.

        name (str):   Module name. Default: 'embedding'.
    """

    dim_node: int
    dim_edge: int
    activation: Union[Callable, str] = 'silu'
    name: str = 'embedding'

    @nn.compact
    def __call__(self, *args, **kwargs):
        raise NotImplementedError