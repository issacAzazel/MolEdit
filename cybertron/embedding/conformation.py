# Code for molecule conformation embedding.

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from flax.linen.initializers import truncated_normal
from typing import Optional, Union, Tuple, Callable
from .embedding import Embedding
from ..model.interaction.molct_interaction import InteractionUnit
from ..common.rbf import get_rbf, RadialBasisFunctions
from ..common.cutoff import get_cutoff, Cutoff
from ..common.filter import get_filter, Filter
from ..common.activation import get_activation

class ConformationEmbedding(nn.Module):
    r"""Conformation embedding module.

    ## Args:

        dim_node (int): Dimension of node vectors. Default: 64.

        dim_edge (int): Dimension of edge vectors. Default: 64.

        num_atom_types (int): Number of atom types. Default: 64.

        num_bond_types (int): Number of bond types. Default: 4.

        is_emb_dis (bool): Whether to embed distance. Default: True.

        is_emb_bond (bool): Whether to embed bond. Default: False.

        is_interaction (bool): Whether to use interaction unit. Default: False.

        dis_self (float): Self distance. Default: 0.04.

        cutoff (float): Cutoff distance. Default: 1.0.

        cutoff_func (str, Cutoff): Cutoff function. Default: 'smooth'.

        num_basis (int): Number of basis. Default: None.

        rbf_runc (str, RadialBasisFunctions): Radial basis function. Default: 'gaussian'.

        atom_filter (str, Filter): Atom filter. Default: None.

        bond_filter (str, Filter): Bond filter. Default: None.

        dis_filter (str, Filter): Distance filter. Default: 'residual'.

        activation (str, Callable): Activation function. Default: 'silu'.

        name (str): Module name. Default: 'conformation_embedding'.
    """

    dim_node: int = 64
    dim_edge: int = 64
    num_atom_types: int = 64
    num_bond_types: int = 4
    is_emb_dis: bool = True
    is_emb_bond: bool = False
    is_interaction: bool = False
    dis_self: float = 0.04
    cutoff: float = 1.0
    cutoff_func: Union[str, Cutoff] = 'smooth'
    num_basis: Optional[int] = None
    rbf_runc: Union[str, RadialBasisFunctions] = 'gaussian'
    atom_filter_name: Union[str, None] = None
    bond_filter_name: Union[str, None] = None
    dis_filter_name: Union[str, None] = 'residual'
    activation: Union[str, Callable] = 'silu'
    name: str = "conformation_embedding"

    def setup(self):

        # atom embedding
        self.atom_embedding = nn.Embed(num_embeddings=self.num_atom_types,
                                       features=self.dim_node,
                                       embedding_init=truncated_normal(stddev=1.0),
                                       name="atom_embedding")
        self.atom_filter = get_filter(cls_name=self.atom_filter_name,
                                      dim_in=self.dim_node,
                                      dim_out=self.dim_node,
                                      activation=self.activation,
                                      name="atom_filter")
        
        # conformation embedding
        if self.is_emb_dis:
            self.cutoff_fn = get_cutoff(cls_name=self.cutoff_func,
                                        cutoff=self.cutoff,)
            self.rbf_fn = get_rbf(cls_name=self.rbf_runc,
                                  r_max=self.cutoff,
                                  num_basis=self.num_basis,)
            self.dis_filter = get_filter(cls_name=self.dis_filter_name,
                                         dim_in=self.rbf_fn.num_basis,
                                         dim_out=self.dim_edge,
                                         activation=self.activation,
                                         name="dis_filter")
        
        # bond embedding
        if self.is_emb_bond:
            self.bond_embedding = nn.Embed(num_embeddings=self.num_bond_types,
                                           features=self.dim_edge,
                                           embedding_init=truncated_normal(stddev=1.0),
                                           name="bond_embedding")
            self.bond_filter = get_filter(cls_name=self.bond_filter_name,
                                          dim_in=self.dim_edge,
                                          dim_out=self.dim_edge,
                                          activation=self.activation,
                                          name="bond_filter")
        
        if self.is_emb_dis and self.is_emb_bond:
            self.interaction = None
            if self.is_interaction:
                self.interaction = InteractionUnit(dim_feature=self.dim_edge,
                                                   n_heads=1,
                                                   activation=self.activation,
                                                   name="emb_interaction",)
        
    def __call__(self,
                 atom_type: jax.Array,
                 bond_type: jax.Array,
                 dist: jax.Array,
                 atom_mask: Optional[jax.Array] = None,
                 bond_mask: Optional[jax.Array] = None,
                 dist_mask: Optional[jax.Array] = None,):
        
        # atom embedding & node_embedding
        if atom_mask is None:
            atom_mask = atom_type > 0
        # (A,) -> (A, F_node)
        node_emb = self.atom_embedding(atom_type)
        # (A, F_node) -> (A, F_node)
        if self.atom_filter is not None:
            node_emb = self.atom_filter(node_emb)      
        node_mask = atom_mask

        # distance embedding
        dis_emb = None
        dis_cutoff = None
        if self.is_emb_dis:

            # set distance matrix
            # (A, A)
            eye_mask = jnp.eye(dist.shape[0], dtype=jnp.bool_)
            # (A, A)
            dist = jnp.where(eye_mask, self.dis_self, dist)

            # embedding distance
            # (A, A, B)
            dis_emb = self.rbf_fn(dist)
            if self.dis_filter is not None:
                # (A, A, B) -> (A, A, F_edge)
                dis_emb = self.dis_filter(dis_emb)
            
            # set dis_cutoff and dist_mask
            # (A, A)
            if self.cutoff_fn is None:
                dis_cutoff = jnp.ones_like(dist)
            else:
                dis_cutoff, dist_mask = self.cutoff_fn(dist, dist_mask)
        
        # distance embedding
        bond_emb = None
        bond_cutoff = None
        if self.is_emb_bond:

            # set bond type & embedding
            bond_emb = self.bond_embedding(bond_type)
            if self.bond_filter is not None:
                bond_emb = self.bond_filter(bond_emb)

            # set bond 
            if bond_mask is None:
                # (A, 1) * (1, A) -> (A, A)
                bond_mask = jnp.logical_and(jnp.expand_dims(atom_mask, axis=1), jnp.expand_dims(atom_mask, axis=0))
                # # optional bond mask
                # bond_mask = bond_type > 0

            # (A, A)
            bond_cutoff = jnp.ones_like(bond_mask)
        
        # edge embedding
        edge_emb = None
        edge_mask = None
        edge_cutoff = None
        if self.is_emb_dis and self.is_emb_bond:

            if self.interaction is not None:
                for _ in range(3):
                    node_emb, _edge_ = self.interaction(node_vec=node_emb,
                                                        edge_vec=bond_emb,
                                                        node_mask=node_mask,
                                                        edge_mask=bond_mask,
                                                        edge_cutoff=bond_cutoff,)
                
                edge_emb = dis_emb
                edge_cutoff = dis_cutoff
                edge_mask = dist_mask
            else:
                edge_emb = dis_emb + bond_emb
                edge_cutoff = dis_cutoff
                edge_mask = dist_mask
        
        elif self.is_emb_dis:
            edge_emb = dis_emb
            edge_mask = dist_mask
            edge_cutoff = dis_cutoff

        elif self.is_emb_bond:
            edge_emb = bond_emb
            edge_mask = bond_mask
            edge_cutoff = bond_cutoff
        else:
            raise ValueError("No edge embedding.")
        
        return node_emb, node_mask, edge_emb, edge_mask, edge_cutoff