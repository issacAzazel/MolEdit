# Code for gaussian radial basis function.

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from typing import Optional, Union, Tuple, List
from .rbf import RadialBasisFunctions

class GaussianBasis(RadialBasisFunctions):
    r"""Gaussian type RBF.
    ## Args:
        r_max (Length):         Maximum distance. Default: 1 nm.

        r_min (Length):         Minimum distance. Default: 0 nm.

        sigma (float):          Simga. Default: 0.03 nm.

        delta (float):          Space interval. Default: 0.016 nm.

        num_basis (int):        Number of basis functions. Defatul: None.

        clip_distance (bool):   Whether to clip the value of distance. Default: False.

        name (str):             Name of the module. Default: "gaussian_basis".
    """

    r_max: float = 1.0
    r_min: float = 0.0
    sigma: float = 0.03
    delta: float = 0.016
    num_basis: Optional[int] = None
    clip_distance: bool = False

    def setup(self):

        # check
        assert self.r_max > self.r_min, "[utils/rbf/GaussianBasis] r_max should be larger than r_min."
        self.r_range = self.r_max - self.r_min

        if self.num_basis is None and self.delta is None:
            raise TypeError('[utils/rbf/GaussianBasis] "num_basis" and "delta" cannot both be "None".')
        if self.num_basis is not None and self.num_basis <= 0:
            raise ValueError('[utils/rbf/GaussianBasis] "num_basis" must be larger than 0.')
        if self.delta is not None and self.delta <= 0:
            raise ValueError('[utils/rbf/GaussianBasis] "delta" must be larger than 0.')
        
        self.coefficient = -0.5 * jnp.reciprocal(jnp.square(self.sigma))

        if self.delta is None and self.num_basis is not None:
            self.offsets = jnp.linspace(self.r_min, self.r_max, self.num_basis)
            # self.delta = self.r_range / (self.num_basis - 1)
        else:
            if self.num_basis is None:
                _num_basis = np.ceil(self.r_range / self.delta) + 1
                _num_basis = int(_num_basis)
                self.offsets = self.r_min + jnp.arange(0, _num_basis) * self.delta
            else:
                self.offsets = self.r_min + jnp.arange(0, self.num_basis) * self.delta
    
    def __call__(self, distance: jnp.ndarray) -> jnp.ndarray:
        r"""Compute gaussian type RBF.

        ## Args: 
            distance (Array):                 Distance matrix. Shape: (A, A).
        ## Returns: 
            radial basis embedding (Array):   Embedding of distance matrix. Shape: (A, A, num_basis).

        """

        if self.clip_distance:
            distance = jnp.clip(distance, self.r_min, self.r_max)
        
        # (..., ) -> (..., 1)
        distance = jnp.expand_dims(distance, axis=-1)
        # (..., 1) - (..., num_basis) -> (..., num_basis)
        diff = distance - self.offsets
        # (..., num_basis) -> (..., num_basis)
        rbf = jnp.exp(self.coefficient * jnp.square(diff))

        return rbf

    def __str__(self) -> str:
        return 'GaussianBasis<>'
