# Basic utils for molct.

import jax
import jax.numpy as jnp
import flax.linen as nn

from jax import Array
from flax.linen.initializers import lecun_normal, truncated_normal, normal, glorot_uniform, zeros_init, ones_init
from typing import Optional, Union
from .activation import get_activation
from .layers.dense import LoRAModulatedDense
from cybertron.modules.basic import Softmax1
from cybertron.modules.basic import ActFuncWrapper
from .adaptive_layernorm import adaLN

from config.global_setup import EnvironConfig
global_setup = EnvironConfig() ### Set Hyper-parameters here
NORM_SMALL = global_setup.norm_small
BF16_FLAG = global_setup.bf16_flag
DROPOUT_FLAG = global_setup.use_dropout
REMAT_FLAG = global_setup.remat_flag

__all__ = [
    "str_to_jax_dtype",
    "softmax_with_mask",
    "PositonalEmbedding",
    "MultiheadAttention",
    "HyperformerPairBlock",
    "HyperformerAtomBlock",
    "FeatureTransformer",
    "HyperAttention",
    "Attention",
    "OuterProduct",
    "Transition",
]

def str_to_jax_dtype(input_str):

    dtype_mapping = {
        'float32': jnp.float32,
        'float16': jnp.float16,
        'bfloat16': jnp.bfloat16,
    }
    
    if input_str in dtype_mapping:
        return dtype_mapping[input_str]
    else:
        raise ValueError("Supported dtypes: float32, float16, bfloat16, but got {}".format(input_str))

def softmax_with_mask(x: jnp.ndarray, 
                      mask: jnp.ndarray, 
                      epsilon: float = -5.0e4) -> jnp.ndarray:

    x = jnp.where(mask, x, epsilon * jnp.ones_like(x))
    return jax.nn.softmax(x, axis=-1)

class PositionalEmbedding(nn.Module):

    dim_feature: int
    use_public_layer_norm: bool = True

    @nn.compact        
    def __call__(self, 
                 x_i: Array,
                 g_ij: Array,
                 time: float = 0.0,
                 ):
        
        # Initializing params
        if self.use_public_layer_norm:
            norm_fn = nn.LayerNorm(name="norm")
            x_norm_fn = norm_fn
            g_norm_fn = norm_fn
        else:
            x_norm_fn = nn.LayerNorm(name="x_norm")
            g_norm_fn = nn.LayerNorm(name="g_norm")
        
        q_gen = nn.Dense(features=self.dim_feature, use_bias=False, name="q_gen")
        k_gen = nn.Dense(features=self.dim_feature, use_bias=False, name="k_gen")
        v_gen = nn.Dense(features=self.dim_feature, use_bias=False, name="v_gen")

        # Do caculation
        # (A, A, F) -> (A, F)
        g_ii = jnp.diagonal(g_ij, axis1=0, axis2=1).swapaxes(0, 1)
        # (A, F) * (A, F) -> (A, F)
        xg_i = x_i * g_ii
        # (A, F) -> (A, 1, F)
        xg_ii = jnp.expand_dims(xg_i, axis=-2)

        # (A, A, F) * (1, A, F) -> (A, A, F)
        xg_ij = jnp.expand_dims(x_i, axis=0) * g_ij

        # (A, 1, F)
        xg_ii = x_norm_fn(xg_ii + time)
        # (A, A, F)
        xg_ij = g_norm_fn(xg_ij + time)
        
        # (A, 1, F) -> (A, 1, F)
        q = q_gen(xg_ii)
        # (A, A, F) -> (A, A, F)
        k = k_gen(xg_ij)
        # (A, A, F) -> (A, A, F)
        v = v_gen(xg_ij)

        return q, k, v

## Liyh: rewrite the MultiheadAttention module with flax
class MultiheadAttention(nn.Module):

    dim_feature: int
    n_heads: int = 1
    
    @nn.compact
    def __call__(self,
                 q_vec: jnp.ndarray,
                 k_mat: jnp.ndarray,
                 v_mat: jnp.ndarray,
                 mask: jnp.ndarray,
                 cutoff: jnp.ndarray,
                 ):
        
        # checking
        assert self.dim_feature % self.n_heads == 0, "[utils/base/MultiheadAttention]: dim_feature must be divisible by n_heads!"
        dim_head = self.dim_feature // self.n_heads
        reshape_tail = (self.n_heads, dim_head)

        # Initializing params
        linear_fn = nn.Dense(features=self.dim_feature,
                             use_bias=False,
                             name="linear_output")
        
        # Do caculation
        # (A, 1, F) -> (A, 1, h, f) -> (A, h, 1, f)
        q_ = jnp.reshape(q_vec, q_vec.shape[:-1] + (self.n_heads, dim_head)).swapaxes(-2, -3)
        # (A, A, F) -> (A, A, h, f) -> (A, h, A, f)
        k_ = jnp.reshape(k_mat, k_mat.shape[:-1] + (self.n_heads, dim_head)).swapaxes(-2, -3)
        v_ = jnp.reshape(v_mat, v_mat.shape[:-1] + (self.n_heads, dim_head)).swapaxes(-2, -3)

        # (A, h, 1, f) @ (A, h, A, f)^T -> (A, h, 1, A)
        att_probs = jnp.einsum("...ij,...kj", q_, k_)
        att_probs = att_probs / jnp.sqrt(dim_head)
        att_probs = softmax_with_mask(att_probs, jnp.expand_dims(jnp.expand_dims(mask, -2), -2))
        att_probs = att_probs * jnp.expand_dims(jnp.expand_dims(cutoff, -2), -2)
        
        # (A, h, 1, A) @ (A, h, A, f) -> (A, h, 1, f)
        att_vec = jnp.einsum("...ij,...jk", att_probs, v_)
        # (A, h, 1, f) -> (A, 1, h, f) -> (A, 1, F)
        att_vec = att_vec.swapaxes(-2, -3).reshape(q_vec.shape)
        att_vec = linear_fn(att_vec)
        
        return att_vec

## Liyh: do the fucking implementation!
class Attention(nn.Module):

    q_data_dim: int
    m_data_dim: int
    output_dim: int
    num_head: int
    gating: bool = False
    sink_attention: bool = False
    key_dim: Optional[int] = None
    value_dim: Optional[int] = None
    fp_type: jnp.dtype = jnp.float32

    def setup(self):
        
        self._dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32

        self._key_dim = self.q_data_dim if self.key_dim is None else self.key_dim
        self._value_dim = self.m_data_dim if self.value_dim is None else self.value_dim
        self._key_dim_per_head = self._key_dim // self.num_head
        self._value_dim_per_head = self._value_dim // self.num_head

        self.q_gen = nn.Dense(features=self._key_dim, use_bias=False, 
                              dtype=self._dtype, param_dtype=jnp.float32, kernel_init=glorot_uniform())
        self.k_gen = nn.Dense(features=self._key_dim, use_bias=False, 
                              dtype=self._dtype, param_dtype=jnp.float32, kernel_init=glorot_uniform())
        self.v_gen = nn.Dense(features=self._value_dim, use_bias=False, 
                              dtype=self._dtype, param_dtype=jnp.float32,
                              kernel_init=glorot_uniform())
        self.linear_output = nn.Dense(features=self.output_dim, use_bias=True, 
                                      dtype=self._dtype, param_dtype=jnp.float32,
                                      kernel_init=glorot_uniform())
        if self.gating:
            self.linear_gating = nn.Dense(features=self._value_dim, use_bias=True, 
                                          dtype=self._dtype, param_dtype=jnp.float32,
                                          kernel_init=zeros_init(), bias_init=ones_init())
            
        if self.sink_attention:
            self.soft_max = Softmax1(-1)
        else:
            self.soft_max = nn.softmax
        
        self.soft_max = ActFuncWrapper(self.soft_max)
        self.sigmoid = ActFuncWrapper(nn.sigmoid)

    def __call__(self, q_data, m_data, bias, pair_bias=None):

        _k = self._key_dim_per_head
        _v = self._value_dim_per_head
        _h = self.num_head
        _aq = q_data.shape[0]
        _am = m_data.shape[0]

        q = self.q_gen(q_data) * _k ** (-0.5)
        k = self.k_gen(m_data)
        v = self.v_gen(m_data)

        q = jnp.reshape(q, (_aq, _h, -1)) # (A, h, c)
        k = jnp.reshape(k, (_am, _h, -1)) # (A', h, c)
        v = jnp.reshape(v, (_am, _h, -1)) # (A', h, c)
        
        # (A, h, c) @ (A', h, c) -> (h, A, A')
        logits = jnp.einsum("ihk,jhk->hij", q, k) ##### bug here
        # logits = jnp.float32(logits) + jnp.float32(bias)

        if pair_bias is not None:
            # logits += jnp.float32(pair_bias)
            logits += pair_bias
        
        # (h, A, A')
        probs = self.soft_max(logits)
        # probs = self.fp_type(probs)

        # (h, A, A') @ (A', h, c) -> (A, h, c)
        weighted_avg = jnp.einsum("hij,jhk->ihk", probs, v) #### bug here

        if self.gating:
            # (A, h*c)
            gating_values = self.linear_gating(q_data)
            # gating_values = jnp.float32(gating_values)
            gating_values = self.sigmoid(gating_values)
            # (A, h, c)
            gating_values = jnp.reshape(gating_values, (_aq, _h, -1))
            # gating_values = self.fp_type(gating_values)
            weighted_avg = weighted_avg * gating_values
        
        # (A, h, c) -> (A, h*c) -> (A, c_out)
        weighted_avg = jnp.reshape(weighted_avg, (_aq, -1))
        output = self.linear_output(weighted_avg)

        return output
    
class HyperAttention(nn.Module):

    atom_act_dim: int
    pair_act_dim: int
    num_head: int
    use_hyper_attention: bool
    ## args in attention
    gating: bool = False
    sink_attention: bool = False
    key_dim: Optional[int] = None
    value_dim: Optional[int] = None
    fp_type: jnp.dtype = jnp.float32
    pre_norm: bool = True

    def setup(self):

        self._dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        
        AttentionUnit = nn.checkpoint(Attention) if REMAT_FLAG else Attention
        
        self.attn_mod = AttentionUnit(q_data_dim=self.atom_act_dim,
                                      m_data_dim=self.atom_act_dim,
                                      output_dim=self.atom_act_dim,
                                      num_head=self.num_head,
                                      gating=self.gating,
                                      sink_attention=self.sink_attention,
                                      key_dim=self.key_dim,
                                      value_dim=self.value_dim,
                                      fp_type=self.fp_type)
        self.query_norm = ActFuncWrapper(nn.LayerNorm(epsilon=NORM_SMALL, dtype=self._dtype, param_dtype=jnp.float32))
        self.feat_2d_norm = ActFuncWrapper(nn.LayerNorm(epsilon=NORM_SMALL, dtype=self._dtype, param_dtype=jnp.float32))
        
        std = 1. / (self.pair_act_dim ** 0.5)
        self.feat_2d_dense = nn.Dense(features=self.num_head, use_bias=False, 
                                      dtype=self._dtype, param_dtype=jnp.float32, kernel_init=normal(std))
        if self.use_hyper_attention:
            self.pair_mat = nn.Dense(features=self.atom_act_dim, use_bias=True, 
                                     dtype=self._dtype, param_dtype=jnp.float32, kernel_init=zeros_init())
            self.mat_norm = ActFuncWrapper(nn.LayerNorm(epsilon=NORM_SMALL, dtype=self._dtype, param_dtype=jnp.float32))
        del std

    def __call__(self, atom_act, pair_act, atom_mask, pair_mask,):

        # (A, A', Cz)
        _a1, _a2, _c = pair_act.shape

        # (A,)
        atom_mask = atom_mask.astype(self._dtype)
        bias = 1e9 * (atom_mask - 1.0) #### bug here
        # (1, 1, A)
        bias = jnp.expand_dims(jnp.expand_dims(bias, 0), 0)

        # (A, Cm)
        atom_act = atom_act.astype(self._dtype)
        if self.pre_norm:
            atom_act = self.query_norm(atom_act)
        # atom_act = self.fp_type(atom_act)

        # (A, A', Cz)
        pair_act = pair_act.astype(self._dtype)
        pair_act = self.feat_2d_norm(pair_act)
        # pair_act = self.fp_type(pair_act)
        # (A, A', Cz) -> (A, A', h) -> (h, A, A')
        pair_bias = self.feat_2d_dense(pair_act)
        pair_bias = jnp.transpose(pair_bias, (2, 0, 1)) ##### bug here

        query_act = atom_act
        key_act = atom_act
        if self.use_hyper_attention:
            # (A, A', Cz) -> (A, A', Cm)
            pair_matrix = self.pair_mat(pair_act) #### zero-init
            pair_matrix = self.mat_norm(pair_matrix)
            # (A, A') -> (A, A', 1)
            pair_mask = jnp.expand_dims(pair_mask, -1)
            # (A, A', Cm) -> (A, Cm)
            query_act += query_act * jnp.sum(pair_matrix * pair_mask, 1) / (jnp.sum(pair_mask, 1) + NORM_SMALL)
            # (A, A', Cm) -> (A', Cm)
            key_act += key_act * jnp.sum(pair_matrix * pair_mask, 0) / (jnp.sum(pair_mask, 0) + NORM_SMALL)

            ### @ZhangJ. Comment for above ###
            '''
            HyperAttention (Efficient):

            qi@(I+Ui)@(I+Vj).T@kj.T = [q@(I+Ui)]@[k@(I+Vj)].T
            I: (A,Cm,Cm); Ui/Vj: (A,Cm,Cm)
            注意: 只有保持Ui(Vj)形式上仅与i(j)有关, 才可以对j(i)执行broadcast (即,affine transform可以absorb到对q或k的改动中).
                所以需要对j进行pooling/reduce操作
            
            Case 1:
            diag(Ui) = mean(Wij,j) or SchNet(qj, Wij; conv. over j), (A,Cm) <=> Ui: (A,Cm,Cm) 且近乎满秩
            diag(Vj) = mean(Wij,i) or SchNet(ki, Wij; conv. over i), (A,Cm) <=> Vj: (A,Cm,Cm) 且近乎满秩

            Case 2:
            Ui_lora = Linear(mean(Wij,j)) 
                    or SchNet(qj, Wij; conv. over j)
            Wij=(A,A',Cm)-> Reduce_j(Mean/Conv.) -> (A,Cm) -> Linear -> (A, Cm*r) -> Reshape -> (A,Cm,r)
            Example: 
                1) Mean Pooling:
                    Pair_act -> Linear -> Wij=(A,A',Cm*r) -> Reduce_Mean(Wij, j) -> (A,Cm*r) -> (A,Cm,r)
                2) Conv. Pooling:
                    Pair_act -> Linear -> Wij=(A,A',Cm*r);
                    qj -> Linear -> (A,Cm*r)
                    Conv(qj, Wij; conv. over j) -> (A,Cm*r) -> (A,Cm,r)

            Ui = Ui_lora@Ui_lora.T = (A,Cm,Cm) 且低秩

            ==>
            q_mod = q@(I+Ui) = q + q@Ui
            k_mod = k@(I+Vj) = k + k@Vj

            --> fast_attention(q_mod, k_mod, v=k_mod, attn_mask)
            '''
        
        atom_act = self.attn_mod(query_act, key_act, bias, pair_bias)
        
        return atom_act

class FeatureTransformer(nn.Module):
    r"""Perform Raw Feature Embedding."""

    atom_types: int
    pair_types: int
    atom_act_dim: int
    pair_act_dim: int
    use_cls: bool
    fp_type: jnp.dtype = jnp.float32

    def setup(self):
        
        self._dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        
        self.preprocess_1d = nn.Dense(features=self.atom_act_dim, 
                                      dtype=self._dtype, param_dtype=jnp.float32, 
                                      kernel_init=lecun_normal())
        self.left_single = nn.Dense(features=self.pair_act_dim, 
                                    dtype=self._dtype, param_dtype=jnp.float32,
                                    kernel_init=lecun_normal())
        self.right_single = nn.Dense(features=self.pair_act_dim, 
                                     dtype=self._dtype, param_dtype=jnp.float32,
                                     kernel_init=lecun_normal())
        self.pair_activations = nn.Dense(features=self.pair_act_dim, 
                                         dtype=self._dtype, param_dtype=jnp.float32,
                                         kernel_init=lecun_normal())
        ### Changed to:
        self.cls_atom_embedding = self.param("cls_atom_embedding", normal(1., jnp.float32), (self.atom_act_dim,))
        self.cls_pair_embedding = self.param("cls_pair_embedding", normal(1., jnp.float32), (self.pair_act_dim,))

    def __call__(self, atom_raw_feat, pair_raw_feat):
        r"""
            atom_raw_feat: (A, C)
            pair_raw_feat: (A, A, C')
        """

        # (A, Cm):
        atom_act = self.preprocess_1d(atom_raw_feat)        
        # (A, A, Cz):
        pair_act = self.pair_activations(pair_raw_feat)
        # (A, Cz):
        left_act = self.left_single(atom_raw_feat)
        right_act = self.right_single(atom_raw_feat)

        # (A, A, Cz):
        pair_act += jnp.expand_dims(left_act, -2) + jnp.expand_dims(right_act, -3)

        if self.use_cls:
            # (A, Cm):
            cls_atom_act = jnp.broadcast_to(self.cls_atom_embedding, atom_act.shape)
            atom_act = jnp.concatenate((cls_atom_act[:1, :], atom_act[:-1, :]), 0)
            # (A, A, Cm):
            cls_pair_act = jnp.broadcast_to(self.cls_pair_embedding, pair_act.shape)
            pair_act = jnp.concatenate((cls_pair_act[:1, :, :], pair_act[:-1, :, :]), 0)
            pair_act = jnp.concatenate((cls_pair_act[:, :1, :], pair_act[:, :-1, :]), 1)

        return atom_act, pair_act

class HyperformerAtomBlock(nn.Module):

    atom_act_dim: int
    pair_act_dim: int
    ## args in hyperattention
    num_head: int
    use_hyper_attention: bool
    gating: bool = False
    sink_attention: bool = False
    key_dim: Optional[int] = None
    value_dim: Optional[int] = None
    fp_type: jnp.dtype = jnp.float32
    ## args in transition
    n_transition: int = 2
    act_fn: str = "relu"
    dropout_rate: float = 0.

    def setup(self):
        
        self._dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32

        self.hyper_attention = HyperAttention(atom_act_dim=self.atom_act_dim,
                                              pair_act_dim=self.pair_act_dim,
                                              num_head=self.num_head,
                                              use_hyper_attention=self.use_hyper_attention,
                                              gating=self.gating,
                                              sink_attention=self.sink_attention,
                                              key_dim=self.key_dim,
                                              value_dim=self.value_dim,
                                              fp_type=self.fp_type)
        TransitionUnit = nn.checkpoint(Transition) if REMAT_FLAG else Transition
        self.transition = TransitionUnit(dim_feature=self.atom_act_dim,
                                         n_transition=self.n_transition,
                                         act_fn=self.act_fn)
        
        self.dropout = nn.Dropout(rate=self.dropout_rate, 
                                  deterministic=not DROPOUT_FLAG)

    def __call__(self, atom_act, pair_act, atom_mask, pair_mask):

        # atom_act = self.fp_type(atom_act)
        # pair_act = self.fp_type(pair_act)
        
        #### attention dropout
        pair_mask = self.dropout(jnp.array(pair_mask, self._dtype))

        ## update atom_act
        atom_act += self.hyper_attention(atom_act, pair_act, atom_mask, pair_mask)
        atom_act += self.transition(atom_act)
        return atom_act
    
class AdaLNHyperformerAtomBlock(nn.Module):

    atom_act_dim: int
    pair_act_dim: int
    ## args in hyperattention
    num_head: int
    use_hyper_attention: bool
    gating: bool = False
    sink_attention: bool = False
    key_dim: Optional[int] = None
    value_dim: Optional[int] = None
    fp_type: jnp.dtype = jnp.float32
    ## args in transition
    n_transition: int = 2
    act_fn: str = "relu"
    dropout_rate: float = 0.

    def setup(self):
        
        self._dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32

        self.hyper_attention = HyperAttention(atom_act_dim=self.atom_act_dim,
                                              pair_act_dim=self.pair_act_dim,
                                              num_head=self.num_head,
                                              use_hyper_attention=self.use_hyper_attention,
                                              gating=self.gating,
                                              sink_attention=self.sink_attention,
                                              key_dim=self.key_dim,
                                              value_dim=self.value_dim,
                                              fp_type=self.fp_type,
                                              pre_norm=False)
        self.adaLN_hyper_attention = adaLN(self.hyper_attention)
        TransitionUnit = nn.checkpoint(Transition) if REMAT_FLAG else Transition
        self.transition = TransitionUnit(dim_feature=self.atom_act_dim,
                                         n_transition=self.n_transition,
                                         act_fn=self.act_fn,
                                         pre_norm=False)
        self.adaLN_transition = adaLN(self.transition)
        
        self.dropout = nn.Dropout(rate=self.dropout_rate, 
                                  deterministic=not DROPOUT_FLAG)

    def __call__(self, atom_act, pair_act, atom_mask, pair_mask, cond):

        # atom_act = self.fp_type(atom_act)
        # pair_act = self.fp_type(pair_act)
        
        #### attention dropout
        pair_mask = self.dropout(jnp.array(pair_mask, self._dtype))

        ## update atom_act
        atom_act = self.adaLN_hyper_attention(atom_act, cond, other_inputs=(pair_act, atom_mask, pair_mask))
        atom_act = self.adaLN_transition(atom_act, cond, other_inputs=())
        return atom_act
    

class HyperformerPairBlock(nn.Module):

    dim_feature: int ## dim_pair_act
    dim_outerproduct: int
    num_transition: int
    act_fn: str = "relu",
    fp_type: jnp.dtype = jnp.float32,
    
    @nn.compact
    def __call__(self, 
                 node_vec: jnp.ndarray, 
                 edge_vec: jnp.ndarray, 
                 node_mask: jnp.ndarray, 
                 edge_mask: Optional[jnp.ndarray]):

        # Initializing params
        OuterProductUnit = nn.checkpoint(OuterProduct) if REMAT_FLAG else OuterProduct
        TransitionUnit = nn.checkpoint(Transition) if REMAT_FLAG else Transition
        outer_product = OuterProductUnit(self.dim_feature, self.dim_outerproduct, self.fp_type)
        transition = TransitionUnit(self.dim_feature, self.num_transition, self.fp_type, self.act_fn)

        edge_vec = jnp.add(edge_vec, outer_product(node_vec, node_mask))
        edge_vec = jnp.add(edge_vec, transition(edge_vec))

        return edge_vec
    
    
class LoRAModulatedHyperformerPairBlock(nn.Module):

    dim_feature: int ## dim_pair_act
    dim_outerproduct: int
    num_transition: int
    act_fn: str = "relu",
    lora_rank: int = 4
    lora_alpha: Union[int, None] = None
    lora_dropout_rate: float = 0.0 
    fp_type: jnp.dtype = jnp.float32,
    
    @nn.compact
    def __call__(self, 
                 node_vec: jnp.ndarray, 
                 edge_vec: jnp.ndarray, 
                 node_mask: jnp.ndarray, 
                 edge_mask: Optional[jnp.ndarray],
                 modulated_params: jnp.ndarray,):

        # Initializing params
        OuterProductUnit = nn.checkpoint(LoRAModulatedOuterProduct) if REMAT_FLAG else LoRAModulatedOuterProduct
        TransitionUnit = nn.checkpoint(LoRAModulatedTransition) if REMAT_FLAG else LoRAModulatedTransition
        outer_product = OuterProductUnit(self.dim_feature, self.dim_outerproduct, 
                                         self.lora_rank, self.lora_alpha, self.lora_dropout_rate,
                                         self.fp_type)
        transition = TransitionUnit(self.dim_feature, self.num_transition, 
                                    self.lora_rank, self.lora_alpha, self.lora_dropout_rate,
                                    self.fp_type, self.act_fn)

        edge_vec = jnp.add(edge_vec, outer_product(node_vec, node_mask, modulated_params))
        edge_vec = jnp.add(edge_vec, transition(edge_vec, modulated_params))

        return edge_vec


class OuterProduct(nn.Module):

    dim_feature: int
    dim_outerproduct: int
    fp_type: jnp.dtype = jnp.float32,
    
    @nn.compact
    def __call__(self, node_vec: jnp.ndarray, node_mask: jnp.ndarray):
        
        _dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32

        left_projection = nn.Dense(features=self.dim_outerproduct,
                                   use_bias=True,
                                   dtype=_dtype,
                                   param_dtype=jnp.float32,
                                   name="left_projection",)
        right_projection = nn.Dense(features=self.dim_outerproduct,
                                    use_bias=True,
                                    dtype=_dtype,
                                    param_dtype=jnp.float32,
                                    name="right_projection",)
        output_projection = nn.Dense(features=self.dim_feature,
                                     use_bias=True,
                                     dtype=_dtype,
                                     param_dtype=jnp.float32,
                                     name="output_projection",)
        
        norm_fn = ActFuncWrapper(nn.LayerNorm(epsilon=NORM_SMALL, dtype=_dtype, param_dtype=jnp.float32, name="norm_fn"))
        
        # Do caculation
        _A = node_vec.shape[0]                          
        _C = self.dim_outerproduct
        act = norm_fn(node_vec)

        # (A, Fm) -> (A, C) -> (C, A) -> (C * A) -> (C * A, 1)
        left_act = left_projection(act)
        left_act = left_act * jnp.expand_dims(node_mask, axis=-1)
        left_act = jnp.transpose(left_act).reshape(-1)
        left_act = jnp.expand_dims(left_act, axis=-1)

        # (A, Fm) -> (A, C) -> (A * C) -> (1, A * C)
        right_act = right_projection(act)
        right_act = right_act * jnp.expand_dims(node_mask, axis=-1)
        right_act = jnp.expand_dims(right_act.reshape(-1), axis=0)

        # (C * A, 1) @ (1, A * C) -> (A * C, A * C)
        out_act = jnp.matmul(left_act, right_act)

        # (C * A, A * C) -> (C, A, A, C) -> (A, A, C, C) -> (A, A, C * C)
        out_act = out_act.reshape((_C, _A, _A, _C))
        out_act = jnp.transpose(out_act, axes=(1, 2, 0, 3))
        out_act = out_act.reshape((_A, _A, _C * _C))

        # (A, A, C * C) -> (A, A, Fz)
        out_act = output_projection(out_act)

        return out_act
    
class LoRAModulatedOuterProduct(nn.Module):

    dim_feature: int
    dim_outerproduct: int
    lora_rank: int = 4
    lora_alpha: Union[int, None] = None
    lora_dropout_rate: float = 0.0 
    fp_type: jnp.dtype = jnp.float32,
    
    @nn.compact
    def __call__(self, node_vec: jnp.ndarray, node_mask: jnp.ndarray, modulated_params: jnp.ndarray):
        
        _dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32          

        left_projection = LoRAModulatedDense(dim_out=self.dim_outerproduct,
                                             use_bias=True,
                                             activation=lambda x:x,
                                             lora_rank=self.lora_rank,
                                             lora_alpha=self.lora_alpha,
                                             lora_dropout_rate=self.lora_dropout_rate,)
                                             #name="left_projection",)
        right_projection = LoRAModulatedDense(dim_out=self.dim_outerproduct,
                                             use_bias=True,
                                             activation=lambda x:x,
                                             lora_rank=self.lora_rank,
                                             lora_alpha=self.lora_alpha,
                                             lora_dropout_rate=self.lora_dropout_rate,)
                                             #name="right_projection",)
        output_projection = LoRAModulatedDense(dim_out=self.dim_feature,
                                               use_bias=True,
                                               activation=lambda x:x,
                                               lora_rank=self.lora_rank,
                                               lora_alpha=self.lora_alpha,
                                               lora_dropout_rate=self.lora_dropout_rate,)
                                               #name="output_projection",)
        
        norm_fn = ActFuncWrapper(nn.LayerNorm(epsilon=NORM_SMALL, dtype=_dtype, param_dtype=jnp.float32,)) #name="norm_fn"))
        
        # Do caculation
        _A = node_vec.shape[0]                          
        _C = self.dim_outerproduct
        act = norm_fn(node_vec)

        # (A, Fm) -> (A, C) -> (C, A) -> (C * A) -> (C * A, 1)
        left_act = left_projection(act, modulated_params)
        left_act = left_act * jnp.expand_dims(node_mask, axis=-1)
        left_act = jnp.transpose(left_act).reshape(-1)
        left_act = jnp.expand_dims(left_act, axis=-1)

        # (A, Fm) -> (A, C) -> (A * C) -> (1, A * C)
        right_act = right_projection(act, modulated_params)
        right_act = right_act * jnp.expand_dims(node_mask, axis=-1)
        right_act = jnp.expand_dims(right_act.reshape(-1), axis=0)

        # (C * A, 1) @ (1, A * C) -> (A * C, A * C)
        out_act = jnp.matmul(left_act, right_act)

        # (C * A, A * C) -> (C, A, A, C) -> (A, A, C, C) -> (A, A, C * C)
        out_act = out_act.reshape((_C, _A, _A, _C))
        out_act = jnp.transpose(out_act, axes=(1, 2, 0, 3))
        out_act = out_act.reshape((_A, _A, _C * _C))

        # (A, A, C * C) -> (A, A, Fz)
        out_act = output_projection(out_act, modulated_params)

        return out_act

class Transition(nn.Module):

    dim_feature: int
    n_transition: int
    fp_type: jnp.dtype = jnp.float32
    act_fn: str = "relu"
    pre_norm: bool = True
    
    @nn.compact
    def __call__(self,
                 edge_vec: jnp.ndarray,
                 ):
        
        _dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        
        # Initializing params
        dim_transition = int(self.dim_feature * self.n_transition)
        norm_fn = ActFuncWrapper(nn.LayerNorm(epsilon=NORM_SMALL, dtype=_dtype, param_dtype=jnp.float32)) if self.pre_norm else lambda x: x
        transition_1 = nn.Dense(features=dim_transition,
                                use_bias=True,
                                name="transition_1",
                                dtype=_dtype,
                                param_dtype=jnp.float32,)
        transition_2 = nn.Dense(features=self.dim_feature,
                                use_bias=True,
                                name="transition_2",
                                dtype=_dtype,
                                param_dtype=jnp.float32,)
        act_fn = get_activation(self.act_fn)

       
        # Do caculation
        # (A, A, F) -> (A, A, F * T)
        act = norm_fn(edge_vec)
        act = transition_1(act)
        # Activation function
        act = act_fn(act)
        # (A, A, F * T) -> (A, A, F)
        act = transition_2(act)

        return act
    
class LoRAModulatedTransition(nn.Module):

    dim_feature: int
    n_transition: int
    lora_rank: int = 4
    lora_alpha: Union[int, None] = None
    lora_dropout_rate: float = 0.0 
    fp_type: jnp.dtype = jnp.float32
    act_fn: str = "relu"
    
    @nn.compact
    def __call__(self,
                 edge_vec: jnp.ndarray,
                 modulated_params: jnp.ndarray,
                 ):
        
        _dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        
        # Initializing params
        dim_transition = int(self.dim_feature * self.n_transition)
        norm_fn = ActFuncWrapper(nn.LayerNorm(epsilon=NORM_SMALL, dtype=_dtype, param_dtype=jnp.float32))
        transition_1 = LoRAModulatedDense(dim_out=dim_transition,
                                          use_bias=True,
                                          activation=lambda x:x,
                                          lora_rank=self.lora_rank,
                                          lora_alpha=self.lora_alpha,
                                          lora_dropout_rate=self.lora_dropout_rate,)
                                          #name="transition_1",)
        transition_2 = LoRAModulatedDense(dim_out=self.dim_feature,
                                          use_bias=True,
                                          activation=lambda x:x,
                                          lora_rank=self.lora_rank,
                                          lora_alpha=self.lora_alpha,
                                          lora_dropout_rate=self.lora_dropout_rate,)
                                          #name="transition_1",)
        
        act_fn = get_activation(self.act_fn)

       
        # Do caculation
        # (A, A, F) -> (A, A, F * T)
        act = norm_fn(edge_vec)
        act = transition_1(act, modulated_params)
        # Activation function
        act = act_fn(act)
        # (A, A, F * T) -> (A, A, F)
        act = transition_2(act, modulated_params)

        return act


# Test the modules
if __name__ == "__main__":
    
    pass