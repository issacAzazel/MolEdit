# Code for smooth cutoff function.

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from typing import Optional, Type, Union, List, Tuple

from .cutoff import Cutoff, _cutoff_register

@_cutoff_register('smooth')
class SmoothCutoff(Cutoff):
    r"""Smooth cutoff network.

    ## Reference:
        Ebert, D. S.; Musgrave, F. K.; Peachey, D.; Perlin, K.; Worley, S.
        Texturing & Modeling: A Procedural Approach; Morgan Kaufmann: 2003

    ## Math:
        r_min < r < r_max:
        f(r) = 1.0 -  6 * ( r / r_cutoff ) ^ 5
                   + 15 * ( r / r_cutoff ) ^ 4
                   - 10 * ( r / r_cutoff ) ^ 3
        r >= r_max: f(r) = 0
        r <= r_min: f(r) = 1

        reverse:
        r_min < r < r_max:
        f(r) =     6 * ( r / r_cutoff ) ^ 5
                - 15 * ( r / r_cutoff ) ^ 4
                + 10 * ( r / r_cutoff ) ^ 3
        r >= r_max: f(r) = 1
        r <= r_min: f(r) = 0

    ## Args:
        cutoff (float): Cutoff distance.

    """

    cutoff: Optional[float]
    name: str = "smooth_cutoff"

    @nn.compact    
    def __call__(self, 
                 distance: jax.Array, 
                 mask: Optional[jax.Array] = None, 
                 cutoff: Optional[jax.Array] = None) -> Tuple[jax.Array, jax.Array]:
        r"""Compute cosine cutoff.
        
        ## Args:
            distance (Distance):    Array of shape (A, A). Distance between atoms.
            mask (Mask):            Array of shape (A, A). Mask for distance.
            cutoff (Cutoff):        Array of shape (A, A). Cutoff distance. Default: None.
        
        ## Returns:
            decay (Array): Array of shape (A, A). Data type is float.
            mask (Array):  Array of shape (A, A). Data type is bool.
        """

        if cutoff is None:
            cutoff = self.cutoff
        
        dis = distance / cutoff
        decay = 1.0 - 6.0 * jnp.power(dis, 5) + 15.0 * jnp.power(dis, 4) - 10.0 * jnp.power(dis, 3)

        mask_upper = distance > 0
        mask_lower = distance < cutoff
        if mask is not None:
            mask_lower = jnp.logical_and(mask_lower, mask)
        else:
            mask_lower = mask_lower
        
        decay = jnp.where(mask_upper, decay, 1)
        decay = jnp.where(mask_lower, decay, 0)

        return decay, mask_lower