"""Transformer variants"""

import jax
import jax.numpy as jnp

from flax import linen as nn
from ..modules.transformer_blocks import Attention, PreNonLinear, PostNonLinear, FeedForwardNet
from dataclasses import field
from config.global_setup import EnvironConfig
global_setup = EnvironConfig() ### Set Hyper-parameters here
REMAT_FLAG = global_setup.remat_flag
BF16_FLAG = global_setup.bf16_flag
SAFE_PRECISION_FLAG = global_setup.safe_precision_flag

class SelfResidualTransformer(nn.Module):

    q_act_dim: int
    pair_act_dim: int
    num_head: int
    intermediate_dim: int
    hidden_dim: int = None
    attn_scale: float = 1.
    ffn_scale: float = 1.
    pre_attention_operation_list: tuple = ("PairEmbedding",) # ["PairEmbedding"] # ["PairEmbedding", "LN", "AttDropout"]
    post_attention_operation_list: tuple = ("Dropout", "LN",) # ["Dropout", "LN"] # ["Dropout", "LN", "ResiDual"]
    post_ffn_operation_list: tuple = ("Dropout",) # ["Dropout"] # ["Dropout", "LN", "ResiDual"]
    dropout_rate: float = 0.
    norm_method: str = "rmsnorm" # ["layernorm", "rmsnorm"]
    gating: bool = True
    sink_attention: bool = False
    init_method: str = "AF2"
    init_sigma: float = 0.02
    swish_beta: float = 1.

    def setup(self):
        
        self._dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        self._safedtype = jnp.float32 if SAFE_PRECISION_FLAG else self._dtype 
        
        if self.hidden_dim is None:
            hidden_dim = self.q_act_dim
        else:
            hidden_dim = self.hidden_dim
        assert hidden_dim % self.num_head == 0

        self.pre_attention = PreNonLinear(
            q_data_dim=self.q_act_dim,
            m_data_dim=self.q_act_dim,
            pair_act_dim=self.pair_act_dim,
            num_head=self.num_head,
            operation_list=self.pre_attention_operation_list,
            dropout_rate=self.dropout_rate,
            norm_method=self.norm_method,
            self_attention=True,
            )
        
        attention_ = nn.checkpoint(Attention) if REMAT_FLAG else Attention
        self.attention = attention_(
            q_data_dim=self.q_act_dim,
            m_data_dim=self.q_act_dim,
            hidden_dim=self.q_act_dim,
            num_head=self.num_head,
            output_dim=self.q_act_dim,
            sink_attention=self.sink_attention,
            gating=self.gating,
            )
        
        self.post_attention = PostNonLinear(
            o_data_dim=self.q_act_dim,
            operation_list=self.post_attention_operation_list,
            dropout_rate=self.dropout_rate,
            norm_method=self.norm_method,
            accumulated_scale=self.attn_scale,
            )
        
        ffn_ = nn.checkpoint(FeedForwardNet) if REMAT_FLAG else FeedForwardNet
        self.ffn = ffn_(
            input_dim=self.q_act_dim,
            intermediate_dim=self.intermediate_dim, ### @ZhangJ. ToBeDone
            output_dim=self.q_act_dim,
            init_method=self.init_method,
            init_sigma=self.init_sigma,
            swish_beta=self.swish_beta,
        )
        
        self.post_ffn = PostNonLinear(
            o_data_dim=self.q_act_dim,
            operation_list=self.post_ffn_operation_list,
            dropout_rate=self.dropout_rate,
            norm_method=self.norm_method,
            accumulated_scale=self.ffn_scale,
            execute_residual=True,
            )

        # if RECOMPUTE_FLAG:
        #     self.attention.recompute()
        #     self.ffn.recompute()

    def __call__(self, act, accumulated_act, attention_masks, pair_act=0., pos_index=None):
        ### Shapes:

        q_act = act
        k_act = act
        v_act = act
        residual_act = act
        q_mask, k_mask_, mask_2d_ = attention_masks

        q_act, k_act, v_act, pair_bias_fp32 = \
            self.pre_attention(q_act, k_act, v_act, attention_masks, pair_act=pair_act)

        attention_output = self.attention(q_act, k_act, v_act, pair_bias_fp32)
        residual_act, accumulated_act = self.post_attention(residual_act, attention_output, q_mask, accumulated_act)

        ffn_output = self.ffn(residual_act)
        residual_act, accumulated_act = self.post_ffn(residual_act, ffn_output, q_mask, accumulated_act)

        return residual_act, accumulated_act