# Initialize the filter module

from typing import Union, Optional
from .filter import Filter, _FILTER_BY_KEY
from .dense import DenseFilter
from .residual import ResFilter

_FILTER_BY_NAME = {filter.__name__: filter for filter in _FILTER_BY_KEY.values()}


def get_filter(cls_name: Union[Filter, str, dict, None],
               dim_in: int,
               dim_out: int,
               activation = None,
               **kwargs,
               ) -> Optional[Filter]:
    """get filter by name"""

    if isinstance(cls_name, Filter):
        return cls_name

    if cls_name is None:
        return None

    if isinstance(cls_name, dict):
        return get_filter(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _FILTER_BY_KEY.keys():
            return _FILTER_BY_KEY[cls_name.lower()](dim_in=dim_in,
                                                    dim_out=dim_out,
                                                    activation=activation,
                                                    **kwargs)
        if cls_name in _FILTER_BY_NAME.keys():
            return _FILTER_BY_NAME[cls_name](dim_in=dim_in,
                                             dim_out=dim_out,
                                             activation=activation,
                                             **kwargs)

        raise ValueError(
            "The filter corresponding to '{}' was not found.".format(cls_name))

    raise TypeError("Unsupported filter type '{}'.".format(type(cls_name)))