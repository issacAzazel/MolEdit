# Basic code for readout module.

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from typing import Optional, Union, Tuple, List, Callable
from ..common.activation import get_activation

_READOUT_BY_KEY = dict()

def _readout_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _READOUT_BY_KEY:
            _READOUT_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _READOUT_BY_KEY:
                _READOUT_BY_KEY[alias] = cls

        return cls

    return alias_reg

class Readout(nn.Module):
    r"""Readout module base class.

    ## Args:

        dim_output (int): Dimension of outputs. Default: 1

        dim_node_rep (int): Dimension of node vectors. Default: None

        dim_edge_rep (int): Dimension of edge vectors. Default: None

        activation (str, Callable): Activation function, Default: None

    ## Symbols:

        B: Batch size.

        A: Number of atoms.

        T: Number of atom types.

        Y: Output dimension.
    """

    dim_output: int = 1
    dim_node_rep: Optional[int] = None
    dim_edge_rep: Optional[int] = None
    activation: Union[str, Callable] = None
    name: str = "readout"

    def setup(self):

        if isinstance(self.activation, str):
            self.act_fn = get_activation(self.activation)
        elif isinstance(self.activation, Callable):
            self.act_fn = self.activation
        else:
            raise ValueError("[readout] Invalid activation type!") 

    def __call__(self,
                 node_rep: jax.Array,
                 edge_rep: jax.Array,
                 node_mask: jax.Array,
                 edge_mask: jax.Array,
                 edge_cutoff: jax.Array,
                 ) -> jax.Array:
        
        r"""Compute readout.

        ## Args:

            node_rep (jax.Array): Node representations, shape (A, F).

            edge_rep (jax.Array): Edge representations, shape (A, A, F).

            node_mask (jax.Array): Node masks, shape (A,).

            edge_mask (jax.Array): Edge masks, shape (A, A).

            edge_cutoff (jax.Array): Edge cutoffs, shape (A, A).
        
        ## Returns:

            output (jax.Array): Readout, shape (Y,), while Y is dim_output.
            
        """
        
        raise NotImplementedError