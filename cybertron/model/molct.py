# Code for the Molct model.
# Reference:
# Zhang, J.; Zhou, Y.; Lei, Y.-K.; Yang, Y. I.; Gao, Y. Q.,
# Molecular CT: unifying geometry and representation learning for molecules at different scales [J/OL].
# arXiv preprint, 2020: arXiv:2012.11816 [2020-12-22]. https://arxiv.org/abs/2012.11816

import jax
import jax.numpy as jnp
import numpy as np

from typing import Optional, Union, Callable
from flax import linen as nn
from .interaction.molct_interaction import InteractionUnit
from ..common.filter import get_filter
from ..common.activation import get_activation

class MolCT(nn.Module):
    r"""Molecular CT model.

    ## Args:

        dim_feature (int): Dimension of node and edge vectors.

        dim_node_emb (int): Dimension of node embeddings. Default: None.

        dim_edge_emb (int): Dimension of edge embeddings. Default: None.

        is_edge_update (bool): Whether to update edge vectors. Default: False.

        is_coupled_interaction (bool): Whether to use coupled interaction. Default: False.

        n_interaction (int): Number of interaction units. Default: 3.

        n_heads (int): Number of heads in multi-head attention. Default: 8.

        dim_outer_pdct (int): Dimension of outer product. Default: 32.

        num_transition (int): Number of transition in hyperformer. Default: 4.

        fp_type (dtype): Floating point type. Default: jnp.float32.

        activation (Callable, str): Activation function, Default: 'silu'.

        name (str): Name of the module. Default: 'molct'.
    
    """

    dim_feature: int
    dim_node_emb: Optional[int] = None
    dim_edge_emb: Optional[int] = None
    is_edge_update: bool = False
    is_coupled_interaction: bool = False
    n_interaction: int = 3
    n_heads: int = 8
    dim_outer_pdct: int = 32
    num_transition: int = 4
    activation: Union[Callable, str] = 'silu'
    name: str = 'molct'

    # def setup(self):

    #     self.act_fn = get_activation(self.activation)

    #     ## using shape inference instead
    #     # build filter
    #     dim_node_emb = self.dim_feature if self.dim_node_emb is None else self.dim_node_emb
    #     dim_edge_emb = self.dim_feature if self.dim_edge_emb is None else self.dim_edge_emb

    #     self.node_filter = None
    #     if self.dim_feature != dim_node_emb:
    #         self.node_filter = get_filter(cls_name='residual',
    #                                       dim_in=dim_node_emb,
    #                                       dim_out=self.dim_feature,
    #                                       activation=self.activation,
    #                                       name="node_filter")
        
    #     self.edge_filter = None
    #     if self.dim_feature != dim_edge_emb:
    #         self.edge_filter = get_filter(cls_name='residual',
    #                                       dim_in=dim_edge_emb,
    #                                       dim_out=self.dim_feature,
    #                                       activation=self.activation,
    #                                       name="edge_filter")
        
    #     # build interaction
    #     self.build_interaction()

    def build_interaction(self):

        if self.is_coupled_interaction:
            interaction = \
                [
                    InteractionUnit(dim_feature=self.dim_feature,
                                    n_heads=self.n_heads,
                                    dim_outer_pdct=self.dim_outer_pdct,
                                    num_transition=self.num_transition,
                                    is_edge_update=self.is_edge_update,
                                    activation=self.activation,
                                    name=f"interaction_unit")
                ] * self.n_interaction
            
        else:
            interaction = \
                [
                    InteractionUnit(dim_feature=self.dim_feature,
                                    n_heads=self.n_heads,
                                    dim_outer_pdct=self.dim_outer_pdct,
                                    num_transition=self.num_transition,
                                    is_edge_update=self.is_edge_update,
                                    activation=self.activation,
                                    name=f"interaction_unit_{_idx}")
                    for _idx in range(self.n_interaction)
                ]
            
        return interaction

    @nn.compact
    def __call__(self,
                 node_emb: jax.Array,
                 node_mask: jax.Array,
                 edge_emb: jax.Array,
                 edge_mask: jax.Array,
                 edge_cutoff: jax.Array,
                 ):
        r"""
        ## Args:

            node_emb (jax.Array): Node embeddings. Shape: (A, F).

            node_mask (jax.Array): Node masks. Shape: (A,).

            edge_emb (jax.Array): Edge embeddings. Shape: (A, A, F).

            edge_mask (jax.Array): Edge masks. Shape: (A, A).

            edge_cutoff (jax.Array): Edge cutoffs. Shape: (A, A).
        
        """

        # build filter & interaction
        # self.build_interaction()
        interaction_stack = self.build_interaction()

        dim_node_emb = node_emb.shape[-1]
        dim_edge_emb = edge_emb.shape[-1]

        node_filter = None
        if self.dim_feature != dim_node_emb:
            node_filter = get_filter(cls_name='residual',
                                     dim_in=dim_node_emb,
                                     dim_out=self.dim_feature,
                                     activation=self.activation,
                                     name="node_filter")
        
        edge_filter = None
        if self.dim_feature != dim_edge_emb:
            edge_filter = get_filter(cls_name='residual',
                                     dim_in=dim_edge_emb,
                                     dim_out=self.dim_feature,
                                     activation=self.activation,
                                     name="edge_filter")
        
        ## do fucking calculation
        # (A, A)
        diagonal_mask = jnp.eye(node_emb.shape[-2], dtype=jnp.bool_)
        # (A, A) or (A, A) -> (A, A)
        edge_mask = jnp.logical_or(edge_mask, diagonal_mask)

        if node_filter is not None:
            node_vec = node_filter(node_emb)
        else:
            node_vec = node_emb

        if edge_filter is not None:
            edge_vec = edge_filter(edge_emb)
        else:
            edge_vec = edge_emb
        
        for interaction_net in interaction_stack:
            node_vec, edge_vec = interaction_net(
                node_vec=node_vec,
                node_mask=node_mask,
                edge_vec=edge_vec,
                edge_mask=edge_mask,
                edge_cutoff=edge_cutoff,
                )
        
        return node_vec, edge_vec
    