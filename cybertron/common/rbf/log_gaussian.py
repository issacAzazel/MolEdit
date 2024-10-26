# Code for log-gaussian radial basis function.

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn 
import math

from typing import Optional, Union, Tuple, List
from .rbf import RadialBasisFunctions, _rbf_register

from config.global_setup import EnvironConfig
global_setup = EnvironConfig() ### Set Hyper-parameters here
NORM_SMALL = global_setup.norm_small
BF16_FLAG = global_setup.bf16_flag

@_rbf_register('log_gaussian')
class LogGaussianBasis(RadialBasisFunctions):
    r"""Log Gaussian type RBF.
    ## Args:
        r_max (Length):         Maximum distance. Defatul: 1 nm.

        r_min (Length):         Minimum distance. Default: 0.04 nm.

        sigma (float):          Simga. Default: 0.3 nm.

        delta (float):          Space interval. Default: 0.0512 nm.

        num_basis (int):        Number of basis functions. Defatul: None.

        rescale (bool):         Whether to rescale the output of RBF from -1 to 1. Default: True

        clip_distance (bool):   Whether to clip the value of distance. Default: False.

        r_ref (float):          Reference distance. Default: 1 nm.

        name (str):             Name of the module. Default: "gaussian_basis".

    """

    r_max: float = 1.0
    r_min: float = 0.04
    sigma: float = 0.3
    delta: float = None
    num_basis: Optional[int] = None
    rescale: bool = True
    clip_distance: bool = False
    r_ref: float = 1.0

    def setup(self):
        
        self._dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        # self._dtype = jnp.float32

        if self.r_max <= self.r_min:
            raise ValueError("[utils/rbf/LogGaussianBasis] r_max should be larger than r_min.")
        
        self.r_range = self.r_max - self.r_min

        if self.num_basis is None and self.delta is None:
            raise TypeError('[utils/rbf/LogGaussianBasis] "num_basis" and "delta" cannot both be "None".')
        if self.num_basis is not None and self.num_basis <= 0:
            raise ValueError('[utils/rbf/LogGaussianBasis] "num_basis" must be larger than 0.')
        if self.delta is not None and self.delta <= 0:
            raise ValueError('[utils/rbf/LogGaussianBasis] "delta" must be larger than 0.')

        self.log_rmin = math.log(self.r_min/self.r_ref)
        self.log_rmax = math.log(self.r_max/self.r_ref)
        self.log_range = self.log_rmax - self.log_rmin
        if self.delta is None and self.num_basis is not None:
            self.offsets = jnp.linspace(self.log_rmin, self.log_rmax, self.num_basis, 
                                        dtype=self._dtype)
            # self.delta = self.log_range / (self.num_basis - 1)
        else:
            if self.num_basis is None:
                _num_basis = int(math.ceil(self.log_range / self.delta)) + 1
                self.offsets = self.log_rmin + jnp.arange(0, _num_basis, dtype=self._dtype) * self.delta
            else:
                self.offsets = self.log_rmin + jnp.arange(0, self.num_basis, dtype=self._dtype) * self.delta
        
        self.coefficient = -0.5 * 1.0 / (math.pow(self.sigma, 2.0))
        self.inv_ref = 1.0 / self.r_ref

    def __call__(self, distance: jnp.ndarray) -> jnp.ndarray:
        """Compute gaussian type RBF.

        ## Args:
            distance (Array): Array of shape `(...)`. Data type is float.

        ## Returns:
            rbf (Array):      Array of shape `(..., K)`. Data type is float.

        """
        if self.clip_distance:
            distance = jnp.clip(distance, self.r_min, self.r_max)
        
        # (...,) -> (..., 1)
        # log_r = jnp.log(distance * self.inv_ref) ## Liyh: The main difference between jax and ms
        log_r = jnp.log(jnp.maximum(distance * self.inv_ref, NORM_SMALL)) #### prevent bugs in BF16
        log_r = jnp.expand_dims(log_r, axis=-1)
        # (..., 1) - (..., K) -> (..., K)
        log_diff = log_r - self.offsets
        rbf = jnp.exp(self.coefficient * jnp.square(log_diff))

        if self.rescale:
            rbf = rbf * 2.0 - 1.0
        
        return rbf
    
    def __str__(self):
        return 'LogGaussianBasis<>'