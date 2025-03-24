# Readout init

import flax.linen as nn
from typing import Optional, Union, Tuple, List, Callable

from .readout import _READOUT_BY_KEY, Readout
from .node import AtomwiseReadout, GraphReadout
from .edge import PairwiseReadout
from .gfn import GFNReadout, GFNScalarReadout
from .adaln_gfn import AdaLNGFNReadout

__all__ = [
    'AtomwiseReadout',
    'GraphReadout',
    'PairwiseReadout',
    'GFNReadout',
    'AdaLNGFNReadout'
    'GFNScalarReadout'
    'get_readout',
]

_READOUT_BY_NAME = {out.__name__: out for out in _READOUT_BY_KEY.values()}


def get_readout(cls_name: Union[Readout, str, nn.Module],
                name: str,
                dim_node_rep: int = None,
                dim_edge_rep: int = None,
                activation: Optional[Union[Callable, str]] = None,
                **kwargs,
                ) -> Union[Readout, nn.Module]:
    """get readout function

    Args:
        readout (str):          Name of readout function. Default: None
        model (MolecularGNN): Molecular model. Default: None
        dim_output (int):       Output dimension. Default: 1
        energy_unit (str):      Energy Unit. Default: None

    """
    if isinstance(cls_name, Readout):
        return cls_name
    if cls_name is None:
        return None

    if isinstance(cls_name, dict):
        return get_readout(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _READOUT_BY_KEY.keys():
            return _READOUT_BY_KEY[cls_name.lower()](
                name=name,
                dim_node_rep=dim_node_rep,
                dim_edge_rep=dim_edge_rep,
                activation=activation,
            )
        if cls_name in _READOUT_BY_NAME.keys():
            return _READOUT_BY_NAME[cls_name](
                name=name,
                dim_node_rep=dim_node_rep,
                dim_edge_rep=dim_edge_rep,
                activation=activation,
            )
        raise ValueError(
            "The Readout corresponding to '{}' was not found.".format(cls_name))
    raise TypeError("Unsupported Readout type '{}'.".format(type(cls_name)))