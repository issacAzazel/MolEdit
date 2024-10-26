"""
Radial basis functions.
"""
import jax
from typing import Optional, Union, List, Tuple
from jax.numpy import ndarray

from .rbf import RadialBasisFunctions, _RBF_BY_KEY
from .gaussian import GaussianBasis
from .log_gaussian import LogGaussianBasis
from .bessel import BesselBasis, NormBesselBasis

__all__ = [
    'RadicalBasisFunctions',
    'GaussianBasis',
    'LogGaussianBasis',
    'BesselBasis',
    'NormBesselBasis',
    'get_rbf'
]

_RBF_BY_NAME = {rbf.__name__: rbf for rbf in _RBF_BY_KEY.values()}

def get_rbf(cls_name: Union[RadialBasisFunctions, str, dict, None],
            r_max: float = 1.0,
            num_basis: Optional[int] = None,
            **kwargs,
            ) -> Union[RadialBasisFunctions, None]:
    """get RBF by name"""

    if isinstance(cls_name, RadialBasisFunctions):
        return cls_name
    if cls_name is None:
        return None

    if isinstance(cls_name, dict):
        return get_rbf(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _RBF_BY_KEY.keys():
            return _RBF_BY_KEY[cls_name.lower()](r_max=r_max,
                                                 num_basis=num_basis,
                                                 **kwargs,
                                                 )
        if cls_name in _RBF_BY_NAME.keys():
            return _RBF_BY_NAME[cls_name](r_max=r_max,
                                          num_basis=num_basis,
                                          **kwargs,
                                          )

        raise ValueError(
            "The RBF corresponding to '{}' was not found.".format(cls_name))

    raise TypeError("Unsupported RBF type '{}'.".format(type(cls_name)))