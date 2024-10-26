# Code for cosine cutoff function.

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from jax.numpy import ndarray
from typing import Optional, Type, Union, List, Tuple

from .cutoff import Cutoff, _cutoff_register

PI = 3.141592653589793238462643383279502884197169399375105820974944592307

@_cutoff_register('cosine')
class CosineCutoff(Cutoff):
    r"""Cutoff Network.
    
    ## Math:
        f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cut}}\right)\right] & r < r_\text{cut} \\
        0 & r \geq r_\text{cut} \\
        \end{cases}
    
    ## Args:
        cutoff (float): Cutoff distance.
    
    """

    cutoff: Optional[float]
    name: str = "cosine_cutoff"
    pi: float = PI

    @nn.compact
    def __call__(self, 
                 distance: ndarray, 
                 mask: Optional[ndarray] = None, 
                 cutoff: Optional[ndarray] = None) -> Tuple[ndarray, ndarray]:
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
        
        decay = 0.5 * (1 + jnp.cos(self.pi * distance / cutoff))

        if mask is None:
            mask = distance < cutoff
        else:
            mask = jnp.logical_and(mask, distance < cutoff)
        
        decay = jnp.where(mask, decay, 0)
        
        return decay, mask