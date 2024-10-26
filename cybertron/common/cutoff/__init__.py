from typing import Union, Optional
from .cutoff import Cutoff, _CUTOFF_BY_KEY
from .cosine import CosineCutoff
from .smooth import SmoothCutoff
from .gaussian import GaussianCutoff, NormalizedGaussianCutoff

__all__ = [
    'Cutoff',
    'CosineCutoff',
    'SmoothCutoff',
    'get_cutoff',
]

_CUTOFF_BY_NAME = {cut.__name__: cut for cut in _CUTOFF_BY_KEY.values()}

def get_cutoff(cls_name: Union[Cutoff, str, dict],
               cutoff: Optional[float] = None,
               **kwargs
               ) -> Cutoff:
    """get cutoff network by name"""
    if cls_name is None:
        return None
    if isinstance(cls_name, Cutoff):
        return cls_name

    if isinstance(cls_name, dict):
        return get_cutoff(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _CUTOFF_BY_KEY.keys():
            return _CUTOFF_BY_KEY[cls_name.lower()](cutoff=cutoff,
                                                    **kwargs)
        if cls_name in _CUTOFF_BY_NAME.keys():
            return _CUTOFF_BY_NAME[cls_name](cutoff=cutoff,
                                             **kwargs)
        raise ValueError(
            "The Cutoff corresponding to '{}' was not found.".format(cls_name))

    raise TypeError("Unsupported Cutoff type '{}'.".format(type(cls_name)))