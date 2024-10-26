# Code for gaussian cutoff function.

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from jax.numpy import ndarray
from typing import Optional, Type, Union, List, Tuple

from .cutoff import Cutoff, _cutoff_register

@_cutoff_register('gaussian')
class GaussianCutoff(Cutoff):
    r"""Gaussian cutoff network.

    ## Args:
        cutoff (float): Cutoff distance.

        sigma (float): Sigma for gaussian function.

        name (str): Name of cutoff function. Default: 'gaussian_cutoff'.
    """

    cutoff: Optional[float]
    sigma: Optional[float]
    name: str = "gaussian_cutoff"

    @nn.compact
    def __call__(self,
                 distance: jax.Array,
                 mask: jax.Array = None,
                 cutoff: jax.Array = None) -> Tuple[jax.Array, jax.Array]:
        r"""Compute gaussian cutoff.
        
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
        
        sigma = self.sigma
        if self.sigma is None:
            sigma = cutoff ## What?
        
        dis = distance - cutoff
        dis2 = dis * dis
        decay = 1. - jnp.exp(-0.5 * dis2 * jnp.reciprocal(jnp.square(sigma)))

        if mask is None:
            mask = distance < cutoff
        else:
            mask = jnp.logical_and(distance < cutoff, mask)

        decay *= mask

        return decay, mask
    

@_cutoff_register('normalized_gaussian')
class NormalizedGaussianCutoff(Cutoff):
    r"""Gaussian cutoff network.

    ## Args:
        cutoff (float): Cutoff distance.

        sigma (float): Sigma for gaussian function.

        name (str): Name of cutoff function. Default: 'gaussian_cutoff'.
    """

    cutoff: Optional[float]
    sigma: Optional[float]
    name: str = "gaussian_cutoff"

    @nn.compact
    def __call__(self,
                 distance: jax.Array,
                 mask: jax.Array = None,
                 cutoff: jax.Array = None) -> Tuple[jax.Array, jax.Array]:
        r"""Compute gaussian cutoff.
        
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
        
        sigma = self.sigma
        if self.sigma is None:
            sigma = cutoff ## What?
        
        dis = distance - cutoff
        dis2 = dis * dis
        decay = 1. - jnp.exp(-0.5 * dis2 * jnp.reciprocal(jnp.square(sigma)))
        
        scale_factor = 1.0 / (1.0 - jnp.exp(-0.5 * cutoff * cutoff * jnp.reciprocal(jnp.square(sigma))))
        decay = decay * scale_factor

        if mask is None:
            mask = distance < cutoff
        else:
            mask = jnp.logical_and(distance < cutoff, mask)

        decay *= mask

        return decay, mask