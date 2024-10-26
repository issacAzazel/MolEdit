# Basic code for cutoff function.

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from typing import Optional, Type, Union, List, Tuple

_CUTOFF_BY_KEY = dict()
def _cutoff_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _CUTOFF_BY_KEY:
            _CUTOFF_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _CUTOFF_BY_KEY:
                _CUTOFF_BY_KEY[alias] = cls

        return cls

    return alias_reg

class Cutoff(nn.Module):
    r"""Cutoff function.

    ## Args:
        cutoff (float):   Cutoff distance.
    
    """

    cutoff: Optional[float]
    name: str = "cutoff"

    def __call__(self,
                 distance: jax.Array,
                 mask: Optional[jax.Array] = None,
                 cutoff: Optional[jax.Array] = None,
                 ) -> Tuple[jax.Array, jax.Array]:
        r"""Compute cutoff.
        
        ## Args:
            distance (Distance):    Array of shape (A, A). Distance between atoms.
            mask (Mask):            Array of shape (A, A). Mask for distance.
            cutoff (Cutoff):        Array of shape (A, A). Cutoff distance. Default: None.
        
        ## Returns:
            decay (Array): Array of shape (A, A). Data type is float.
            mask (Array):  Array of shape (A, A). Data type is bool.
        """

        raise NotImplementedError