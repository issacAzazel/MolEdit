# Code for bessel radial basis function.

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from typing import Optional, Union, Tuple, List
from flax.linen.initializers import constant
from .rbf import RadialBasisFunctions, _rbf_register
import math

from config.global_setup import EnvironConfig
global_setup = EnvironConfig() ### Set Hyper-parameters here
NORM_SMALL = global_setup.norm_small
BF16_FLAG = global_setup.bf16_flag

@_rbf_register('bessel')
class BesselBasis(RadialBasisFunctions):
    r"""Bessel type RBF.
    ## Args:
        r_max (Length):         Maximum distance.

        num_basis (int):        Number of basis functions. Defatul: None.

        name (str):             Name of the module. Default: "bessel_basis".

    """
    r_max: float
    num_basis: int = 8
    trainable: bool = True
    tol: float = 1e-3
    name: str = "bessel_basis"

    @nn.compact
    def __call__(self, distance: jax.Array) -> jax.Array:
        
        _dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        # _dtype = jnp.float32
        
        d_min = math.sqrt(self.tol * 6.0)

        assert self.r_max != 0, "[utils/rbf/BesselBasis] r_max should not be 0!"
        prefactor = 2.0 / self.r_max
        # prefactor = self.r_max / 2.0

        bessel_weights = jnp.linspace(1.0, self.num_basis, self.num_basis,
                                      dtype=jnp.float32) * jnp.pi
        if self.trainable:
            bessel_weights = self.param("bessel_weights", constant(bessel_weights), (self.num_basis,))
        else:
            bessel_weights = self.param("bessel_weights", constant(bessel_weights), (self.num_basis,))
            bessel_weights = jax.lax.stop_gradient(bessel_weights)
        
        # (..., ) -> (..., 1)
        distance = jnp.expand_dims(distance, axis=-1)
        # (..., 1) -> (..., num_basis)
        bessel_distance = bessel_weights.astype(_dtype) * distance
        bessel_distance = bessel_distance / self.r_max
        alpha = bessel_weights / self.r_max
        numerator = jnp.sin(bessel_distance)
        ret = jnp.where(distance < d_min, 
                        alpha*(1.0 - jnp.square(alpha*distance)/6.0),
                        numerator/jnp.maximum(distance, d_min))
        # ret = prefactor * (numerator / distance)
        ret = prefactor * ret

        return ret
    
@_rbf_register('norm_bessel')
class NormBesselBasis(BesselBasis):
    r"""Normalized Bessel type RBF."""

    r_max: float
    num_basis: int = 8
    trainable: bool = True
    norm_num: int = 4000
    name: str = "bessel_basis"

    @nn.compact
    def __call__(self, distance: jax.Array) -> jax.Array:

        rs = jnp.linspace(0, self.r_max, self.norm_num + 1)
        bs = super().__call__(rs)
        basis_mean = jnp.mean(bs, axis=0)
        basis_std = jnp.std(bs, axis=0)

        # calculate
        basis_dis = super().__call__(distance)
        ret = (basis_dis - basis_mean) / basis_std

        return ret