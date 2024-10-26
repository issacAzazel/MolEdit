# Code for radial basis functions.

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from typing import Optional

_RBF_BY_KEY = dict()

def _rbf_register(*aliases):
    """Return the alias register"""
    def alias_reg(cls):
        name = cls.__name__.lower()
        if name not in _RBF_BY_KEY:
            _RBF_BY_KEY[name] = cls
        for alias in aliases:
            if alias not in _RBF_BY_KEY:
                _RBF_BY_KEY[alias] = cls
        return cls
    return alias_reg

class RadialBasisFunctions(nn.Module):
    r"""Network for radial basis functions.

    ## Args:

        r_max (float):          Maximum distance. Default: 1.0 nm.

        r_min (float):          Minimum distance. Default: 0.0 nm.

        num_basis (int):        Number of basis functions. Default: None.

        clip_distance (bool):   Whether to clip the value of distance. Default: False.
    
    """

    r_max: float = 1.0
    r_min: float = 0.0
    num_basis: Optional[int] = None
    clip_distance: bool = False
    
    def __call__(self, distance: jnp.ndarray) -> jnp.ndarray:
        r"""Compute the radial basis functions.

        ## Args: 
            distance (Array):                 Distance matrix. Shape: (A, A).
        ## Returns: 
            radial basis embedding (Array):   Embedding of distance matrix. Shape: (A, A, num_basis).

        """
        raise NotImplementedError

    def __str__(self):
        return 'RadicalBasisFunctions<>'  
