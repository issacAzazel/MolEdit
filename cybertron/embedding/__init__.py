# Embedding Init

from typing import Union

from .embedding import Embedding
# from .graph import GraphEmbedding ## Liyh: need to be implemented
from .conformation import ConformationEmbedding

__all__ = [
    'GraphEmbedding',
    'Embedding',
    'ConformationEmbedding',
    'get_embedding',
]


