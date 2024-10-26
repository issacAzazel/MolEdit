# Decoder init

import jax.nn as nn
from typing import Optional, Union, Tuple, List, Callable

from flax import linen as nn
from .decoder import Decoder, _DECODER_BY_KEY
from .halve import HalveDecoder
from .residual import ResDecoder

__all__ = [
    'Decoder',
    'HalveDecoder',
    'ResDecoder',
    'get_decoder',
]


_DECODER_BY_NAME = {
    decoder.__name__: decoder for decoder in _DECODER_BY_KEY.values()}


def get_decoder(cls_name: Union[dict, Decoder, str],
                dim_in: int,
                dim_out: int,
                name: str,
                activation: Union[Callable, str] = nn.silu,
                n_layers: int = 1,
                **kwargs
                ) -> Optional[Decoder]:
    """get decoder by name"""
    if cls_name is None or isinstance(cls_name, Decoder):
        return cls_name

    if isinstance(cls_name, dict):
        return get_decoder(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _DECODER_BY_KEY.keys():
            return _DECODER_BY_KEY[cls_name.lower()](
                dim_in=dim_in,
                dim_out=dim_out,
                activation=activation,
                n_layers=n_layers,
                name=name,
                **kwargs
            )
        if cls_name in _DECODER_BY_NAME.keys():
            return _DECODER_BY_NAME[cls_name](
                dim_in=dim_in,
                dim_out=dim_out,
                activation=activation,
                n_layers=n_layers,
                name=name,
                **kwargs
            )

        raise ValueError(
            "The Decoder corresponding to '{}' was not found.".format(cls_name))

    raise TypeError("Unsupported init type '{}'.".format(type(cls_name)))
