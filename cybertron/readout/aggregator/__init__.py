# Aggregator init

from typing import Optional, Union, Tuple, List
from flax import linen as nn
from .node import *
from .edge import *
from .node import _AGGREGATOR_BY_KEY
__all__ = []
__all__.extend(node.__all__)
__all__.extend(edge.__all__)

_AGGREGATOR_BY_NAME = {
    agg.__name__: agg for agg in _AGGREGATOR_BY_KEY.values()}


def get_aggregator(cls_name: Union[nn.Module, str, dict],
                   axis: Union[int, Tuple],
                   name: str,
                   **kwargs,
                   ) -> Union[nn.Module, None]:
    """get aggregator by name"""
    if cls_name is None or isinstance(cls_name, nn.Module):
        return cls_name
    if isinstance(cls_name, dict):
        return get_aggregator(**cls_name)
    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _AGGREGATOR_BY_KEY.keys():
            return _AGGREGATOR_BY_KEY[cls_name.lower()](reduce_axis=axis, name=name, **kwargs)
        if cls_name in _AGGREGATOR_BY_NAME.keys():
            return _AGGREGATOR_BY_NAME[cls_name](reduce_axis=axis, name=name, **kwargs)
        raise ValueError(
            "The Aggregator corresponding to '{}' was not found.".format(cls_name))
    raise TypeError(
        "Unsupported Aggregator type '{}'.".format(type(cls_name)))