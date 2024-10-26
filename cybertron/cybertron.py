# The main program for cybertron. The net will be defined here.
# 2023-10-23 
# Orchestra

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from typing import Union, Tuple, List, Optional
from jax.numpy import ndarray as ndarray

from .embedding import Embedding
from .readout import Readout

class Cybertron(nn.Module):
    r"""The main program for cybertron.

    ## Args:
        embedding: The embedding module.

        model: The interaction module.

        readout: The readout module.
    
    """

    embedding: nn.Module
    model: nn.Module
    readout: Union[Readout, nn.Module, List[Readout], List[nn.Module]]


    @nn.compact        
    def __call__(self,
                 atom_type: ndarray,
                 atom_mask: Optional[ndarray],
                 bond_type: Optional[ndarray],
                 bond_mask: Optional[ndarray],
                 coord: Optional[ndarray],
                 dist_mask: Optional[ndarray],
                 ):

        def get_dist_mat(position, 
                         mask,
                         large_dis):

            # (A, 1, 3) - (1, A, 3) -> (A, A, 3)
            vectors = jnp.expand_dims(position, axis=1) - jnp.expand_dims(position, axis=0)
            large_dis_mat = jnp.where(mask, 0.0, large_dis)
            # (A, A, 3) + (A, A, 1) -> (A, A, 3)
            vectors += jnp.expand_dims(large_dis_mat, axis=-1) # type: ignore
            # (A, A, 3) -> (A, A)
            dist_mat = jnp.linalg.norm(vectors, axis=-1)

            return dist_mat

        if atom_mask is None:
            atom_mask = atom_type > 0
        
        if dist_mask is None:
            # (A, 1) & (1, A) -> (A, A)
            dist_mask = jnp.logical_and(jnp.expand_dims(atom_mask, axis=1), jnp.expand_dims(atom_mask, axis=0))
            # (A, A)
            eye_mask = jnp.logical_not(jnp.eye(atom_mask.shape[0], dtype=jnp.bool_))
            # (A, A) & (A, A) -> (A, A)
            dist_mask = jnp.logical_and(dist_mask, eye_mask)
        
        else:
            # (A, A)
            eye_mask = jnp.logical_not(jnp.eye(atom_mask.shape[0], dtype=jnp.bool_))
            # (A, A) & (A, A) -> (A, A)
            dist_mask = jnp.logical_and(dist_mask, eye_mask)

        # Caculate distance matrix
        if coord is not None:
            large_dis = self.embedding.cutoff * 10.0
            # (A, 3) -> (A, A)
            dist_mat = get_dist_mat(coord, dist_mask, large_dis)
        else:
            dist_mat = None
        
        node_emb, node_mask, edge_emb, edge_mask, edge_cutoff = \
            self.embedding(atom_type, bond_type, dist_mat, atom_mask, bond_mask, dist_mask)
        
        node_vec, edge_vec = \
            self.model(node_emb, node_mask, edge_emb, edge_mask, edge_cutoff)
        
        # modified for water ccsd
        if bond_mask is not None:
            edge_mask = jnp.logical_and(edge_mask, bond_mask)
        
        outputs = ()
        for readout_ in self.readout:
            output = readout_(node_vec=node_vec,
                              edge_vec=edge_vec,
                              node_mask=node_mask,
                              edge_mask=edge_mask,
                              edge_cutoff=edge_cutoff,)
            
            outputs += (output,)
        
        return outputs