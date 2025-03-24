# Code for Neural Interaction Unit (NIU).
# Reference:
# Zhang, J.; Zhou, Y.; Lei, Y.-K.; Yang, Y. I.; Gao, Y. Q.,
# Molecular CT: unifying geometry and representation learning for molecules at different scales [J/OL].
# arXiv preprint, 2020: arXiv:2012.11816 [2020-12-22]. https://arxiv.org/abs/2012.11816

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from typing import Optional, Union, Tuple, Callable
from ...common.base import PositionalEmbedding, MultiheadAttention, HyperformerPairBlock, HyperformerAtomBlock, AdaLNHyperformerAtomBlock, str_to_jax_dtype
from ...common.activation import get_activation
from ...common.config_load import Config

class InteractionUnit(nn.Module):
    r"""Interaction unit for MolCT.

    ## Args:

        dim_feature (int): Dimension of node and edge vectors.

        n_heads (int): Number of heads in multi-head attention.

        dim_outer_pdct (int): Dimension of outer product. Default: 32.
 
        num_transition (int): Number of transition in hyperformer. Default: 4.

        is_edge_update (bool): Whether to update edge vectors. Default: False.

        activation (Callable, str): Activation function, Default: 'silu'.

        fp_type (dtype): Floating point type. Default: jnp.float32.

        name (str): Name of the module. Default: 'neural_interaction_unit'.
    
    """

    dim_feature: int
    n_heads: int
    dim_outer_pdct: int = 32
    num_transition: int = 4
    is_edge_update: bool = False
    activation: Union[Callable, str] = 'silu'
    name: str = 'neural_interaction_unit'

    def setup(self):

        self.act_fn = get_activation(self.activation)

        self.positional_embedding = PositionalEmbedding(dim_feature=self.dim_feature)
        self.multi_head_attention = MultiheadAttention(dim_feature=self.dim_feature, 
                                                       n_heads=self.n_heads)

        self.hyperformer_pair_block = None
        if self.is_edge_update:
            self.hyperformer_pair_block = HyperformerPairBlock(self.dim_feature, 
                                                               self.dim_outer_pdct, 
                                                               self.num_transition)

        self.node_norm = nn.LayerNorm(name="node_norm")
        self.edge_norm = nn.LayerNorm(name="edge_norm")
        
    def __call__(self,
                 node_vec: jax.Array,
                 node_mask: jax.Array,
                 edge_vec: jax.Array,
                 edge_mask: jax.Array,
                 edge_cutoff: jax.Array,
                 ):
        r"""
        ## Args:

            node_vec (jax.Array): Node vectors. Shape: (A, F).

            node_mask (jax.Array): Node masks. Shape: (A,).

            edge_vec (jax.Array): Edge vectors. Shape: (A, A, F).

            edge_mask (jax.Array): Edge masks. Shape: (A, A).

            edge_cutoff (jax.Array): Edge cutoffs. Shape: (A, A).
        """
        
        # Initializing the encoders for nodes and edges.
        def _node_encoder(node_vec: jax.Array,
                          edge_vec: jax.Array,
                          edge_mask: jax.Array,
                          edge_cutoff: jax.Array,
                          time_signal: float = 0.0,
                          ) -> jax.Array:
            
            query, key, value = self.positional_embedding(node_vec,
                                                          edge_vec,
                                                          time_signal,)
            delta_node_vec = self.multi_head_attention(query,
                                                       key,
                                                       value,
                                                       edge_mask,
                                                       edge_cutoff,)
            node_new = node_vec + delta_node_vec.squeeze(axis=1)
            node_new = self.node_norm(node_new)

            return node_new

        def _edge_encoder(node_vec: jax.Array,
                          edge_vec: jax.Array,
                          node_mask: jax.Array,
                          edge_mask: jax.Array,
                          ) -> jax.Array:
            
            edge_vec = self.hyperformer_pair_block(node_vec=node_vec,
                                                   edge_vec=edge_vec,
                                                   node_mask=node_mask,
                                                   edge_mask=edge_mask,) # type: ignore
            edge_vec = self.edge_norm(edge_vec)

            return edge_vec

        if self.is_edge_update:
            node_new = _node_encoder(
                        node_vec=node_vec,
                        edge_vec=edge_vec,
                        edge_mask=edge_mask,
                        edge_cutoff=edge_cutoff,)

            edge_new = _edge_encoder(
                        node_vec=node_vec,
                        edge_vec=edge_vec,
                        node_mask=node_mask,
                        edge_mask=edge_mask,)
        else:
            node_new = _node_encoder(
                        node_vec=node_vec,
                        edge_vec=edge_vec,
                        edge_mask=edge_mask,
                        edge_cutoff=edge_cutoff,)

            edge_new = edge_vec

        return node_new, edge_new                                   
            
class TopoInteractionUnit(nn.Module):

    config: Config

    def setup(self):
        
        ## extract args
        self.atom_act_dim = self.config.atom_act_dim
        self.pair_act_dim = self.config.pair_act_dim
        self.cycles = self.config.cycles

        atom_config = self.config.atom_block
        pair_config = self.config.pair_block

        atom_fp_type = str_to_jax_dtype(atom_config.fp_type)
        self.niu_atom_block = HyperformerAtomBlock(atom_act_dim=self.atom_act_dim,
                                                   pair_act_dim=self.pair_act_dim,
                                                   num_head=atom_config.num_head,
                                                   use_hyper_attention=atom_config.use_hyper_attention,
                                                   gating=atom_config.gating,
                                                   sink_attention=atom_config.sink_attention,
                                                   key_dim=atom_config.key_dim if "key_dim" in atom_config.__dict__.keys() else None,
                                                   value_dim=atom_config.value_dim if "value_dim" in atom_config.__dict__.keys() else None,
                                                   n_transition=atom_config.n_transition,
                                                   act_fn=atom_config.act_fn,
                                                   fp_type=atom_fp_type,
                                                   dropout_rate=atom_config.dropout_rate)
        
        pair_fp_type = str_to_jax_dtype(pair_config.fp_type)
        self.niu_pair_block = HyperformerPairBlock(dim_feature=self.pair_act_dim,
                                                   dim_outerproduct=pair_config.dim_outer_pdct,
                                                   num_transition=pair_config.num_transition,
                                                   act_fn=pair_config.act_fn,
                                                   fp_type=pair_fp_type,)

    def __call__(self, atom_act, pair_act, atom_mask, pair_mask):
        
        ### 1. Update Pair Rep.:
        pair_act = self.niu_pair_block(atom_act, pair_act, atom_mask, pair_mask)

        ### 2. Update MSA Rep.:
        for _ in range(self.cycles):
            atom_act = self.niu_atom_block(atom_act, pair_act, atom_mask, pair_mask)
            
        return atom_act, pair_act


class AdaLNTopoInteractionUnit(nn.Module):

    config: Config

    def setup(self):
        
        ## extract args
        self.atom_act_dim = self.config.atom_act_dim
        self.pair_act_dim = self.config.pair_act_dim
        self.cycles = self.config.cycles

        atom_config = self.config.atom_block
        pair_config = self.config.pair_block

        atom_fp_type = str_to_jax_dtype(atom_config.fp_type)
        self.niu_atom_block = AdaLNHyperformerAtomBlock(atom_act_dim=self.atom_act_dim,
                                                        pair_act_dim=self.pair_act_dim,
                                                        num_head=atom_config.num_head,
                                                        use_hyper_attention=atom_config.use_hyper_attention,
                                                        gating=atom_config.gating,
                                                        sink_attention=atom_config.sink_attention,
                                                        key_dim=atom_config.key_dim if "key_dim" in atom_config.__dict__.keys() else None,
                                                        value_dim=atom_config.value_dim if "value_dim" in atom_config.__dict__.keys() else None,
                                                        n_transition=atom_config.n_transition,
                                                        act_fn=atom_config.act_fn,
                                                        fp_type=atom_fp_type,
                                                        dropout_rate=atom_config.dropout_rate)
        
        pair_fp_type = str_to_jax_dtype(pair_config.fp_type)
        self.niu_pair_block = HyperformerPairBlock(dim_feature=self.pair_act_dim,
                                                   dim_outerproduct=pair_config.dim_outer_pdct,
                                                   num_transition=pair_config.num_transition,
                                                   act_fn=pair_config.act_fn,
                                                   fp_type=pair_fp_type,)

    def __call__(self, atom_act, pair_act, atom_mask, pair_mask, cond):
        
        ### 1. Update Pair Rep.:
        pair_act = self.niu_pair_block(atom_act, pair_act, atom_mask, pair_mask)

        ### 2. Update MSA Rep.:
        for _ in range(self.cycles):
            atom_act = self.niu_atom_block(atom_act, pair_act, atom_mask, pair_mask, cond)
            
        return atom_act, pair_act
