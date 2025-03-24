# Basic code for schnet interaction

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from typing import Optional, Union, Tuple, Callable
from ...common.activation import get_activation
from ...common.filter import DenseFilter
from ...common.layers.mlp import MLP, LoRAModulatedMLP
from ...common.layers.dense import Dense, LoRAModulatedDense

from config.global_setup import EnvironConfig
global_setup = EnvironConfig() ### Set Hyper-parameters here
NORM_SMALL = global_setup.norm_small
BF16_FLAG = global_setup.bf16_flag

class SchnetInteraction(nn.Module):
    r"""Interaction layer of Schnet.

    ## Args:
        dim_filter (int): The dimension of the filter.
        
        activation (Callable, str): The activation function used.

        normalize_filter (bool): Whether to normalize the filter.

    """

    dim_filter: int
    activation: Union[Callable, str] = 'silu'
    normalize_filter: bool = False

    @nn.compact
    def __call__(self, node_vec, node_mask, edge_vec, edge_mask, edge_cutoff):
        r"""
        
        """
        dim_edge_rep = edge_vec.shape[-1]
        dim_node_rep = node_vec.shape[-1]
        dim_filter = self.dim_filter if self.dim_filter is not None else dim_edge_rep

        filter_net = DenseFilter(dim_in=dim_edge_rep, dim_out=dim_filter, activation=self.activation)
        atomwise_bc = nn.Dense(features=dim_filter)
        atomwise_ac = MLP(output_sizes=(dim_node_rep, dim_node_rep), activation=self.activation, activate_final=False)
        
        def _aggregate(inputs, mask):

            if mask is not None:
                # (A, A, W) * (A, A, 1)
                inputs = inputs * jnp.expand_dims(mask, -1)
            _out = jnp.sum(inputs, axis=-2)

            if self.normalize_filter:
                if mask is not None:
                    num = jnp.sum(mask, axis=-2)
                    num = jnp.maximum(num, 1) ## Liyh: need to check this
                else:
                    num = inputs.shape[-2]
                _out = _out / num
            
            return _out
            
        # (A, F) -> (A, W)
        x_i = atomwise_bc(node_vec)
        # (A, A, K) -> (A, A, W)
        g_ij = filter_net(edge_vec)
        # (A, A, W) * (A, A, 1) -> (A, A, W)
        w_ij = g_ij * jnp.expand_dims(edge_cutoff, -1)
        # (1, A, W) * (A, A, W)
        y = jnp.expand_dims(x_i, -3) * w_ij
        # (A, A, W) -> (A, W)
        y = _aggregate(y, edge_mask)
        # (A, W) -> (A, F)
        y = atomwise_ac(y)
        # (A, F) + (A, F) -> (A, F)
        node_new = node_vec + y

        return node_new, edge_vec

# A modified version of SchnetInteraction
class HyperSchnetInteraction(nn.Module):
    r"""Interaction layer of Schnet.

    ## Args:
        dim_filter (int): The dimension of the filter.

        n_filter_hidden (int): The number of hidden layers in the filter.

        activation (Callable, str): The activation function used.

        filter_activation (Callable, str): The activation function used in the filter.

        normalize_filter (bool): Whether to normalize the filter.
    """

    dim_filter: int
    n_filter_hidden: int
    activation: Union[Callable, str] = 'ssp'
    filter_activation: Union[Callable, str] = 'ssp'
    normalize_filter: bool = False

    @nn.compact
    def __call__(self, x, f_ij, c_ij, mask,):

        dim_feature = x.shape[-1]
        
        atomwise_bc = Dense(self.dim_filter, activation=self.activation)
        atomwise_ac = MLP(output_sizes=(dim_feature, dim_feature), activation=self.activation, activate_final=True)
        dis_filter = MLP(output_sizes=(self.dim_filter,) * (self.n_filter_hidden + 1), activate_final=True)

        def _aggregate(inputs, mask):

            if mask is not None:
                # (A, A, W) * (A, A, 1)
                inputs = inputs * jnp.expand_dims(mask, -1)
            _out = jnp.sum(inputs, axis=-2)

            if self.normalize_filter:
                if mask is not None:
                    num = jnp.sum(mask, axis=-2)
                    num = jnp.maximum(num, 1) ## Liyh: need to check this
                else:
                    num = inputs.shape[-2]
                _out = _out / num
            
            return _out
        
        # (A, F) -> (A, W)
        x_i = atomwise_bc(x)
        # (A, A, K) -> (A, A, W)
        f_ij = dis_filter(f_ij)
        # (A, A, W) * (A, A, 1) -> (A, A, W)
        w_ij = f_ij * jnp.expand_dims(c_ij, -1)
        # (A, W) -> (1, A, W)
        x_ij = jnp.expand_dims(x_i, -3) ## Liyh: the right version
        # (1, A, W) * (A, A, W) -> (A, A, W)
        y = x_ij * w_ij
        # (A, A, W) -> (A, W)
        y = _aggregate(y, mask)
        # (A, W) -> (A, F)
        y = atomwise_ac(y)
        # (A, F) + (A, F) -> (A, F)
        x_new = x + y

        return x_new
    
    
class LoRAModulatedSchnetInteraction(nn.Module):
    r"""Interaction layer of Schnet.

    ## Args:
        dim_filter (int): The dimension of the filter.

        n_filter_hidden (int): The number of hidden layers in the filter.

        activation (Callable, str): The activation function used.

        filter_activation (Callable, str): The activation function used in the filter.

        normalize_filter (bool): Whether to normalize the filter.
    """

    dim_filter: int
    n_filter_hidden: int
    activation: Union[Callable, str] = 'ssp'
    filter_activation: Union[Callable, str] = 'ssp'
    normalize_filter: bool = False
    dropout_rate: float = 0.0
    lora_rank: int = 4
    lora_alpha: Union[int, None] = None
    lora_dropout_rate: float = 0.0
    residual: bool = True

    @nn.compact
    def __call__(self, x, f_ij, c_ij, mask, modulated_params):
        
        _dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        
        dim_feature = x.shape[-1]
        atomwise_bc = LoRAModulatedDense(
            dim_out = self.dim_filter, 
            use_bias = True, 
            activation = self.activation, 
            lora_rank = self.lora_rank,
            lora_alpha = self.lora_alpha,
            lora_dropout_rate = self.lora_dropout_rate
        )
        atomwise_ac = LoRAModulatedMLP(
            output_sizes = (dim_feature, dim_feature), 
            with_bias = True, 
            activation = self.activation,
            activate_final = True, 
            dropout_rate = self.dropout_rate, 
            lora_rank  = self.lora_rank,
            lora_alpha = self.lora_alpha, 
            lora_dropout_rate = self.lora_dropout_rate, 
        )
        
        dis_filter = LoRAModulatedMLP( 
            output_sizes = (self.dim_filter,) * (self.n_filter_hidden + 1), 
            with_bias = True, 
            activation = self.activation,
            activate_final = True, 
            dropout_rate = self.dropout_rate, 
            lora_rank = self.lora_rank, 
            lora_alpha = self.lora_alpha, 
            lora_dropout_rate = self.lora_dropout_rate
        )

        def _aggregate(inputs, mask):

            if mask is not None:
                # (A, A, W) * (A, A, 1)
                inputs = inputs * jnp.expand_dims(mask, -1)
            _out = jnp.sum(inputs, axis=-2)

            if self.normalize_filter:
                if mask is not None:
                    # num = jnp.sum(mask, axis=-2)
                    num = jnp.sum(mask, axis=-1, keepdims=True)
                    num = jnp.maximum(num, 1) ## Liyh: need to check this
                else:
                    num = inputs.shape[-2]
                _out = _out / num
            
            return _out
        
        # (A, F) -> (A, W)
        x_i = atomwise_bc(x, modulated_params)
        # (A, A, K) -> (A, A, W)
        f_ij = dis_filter(f_ij, modulated_params)
        # (A, A, W) * (A, A, 1) -> (A, A, W)
        w_ij = f_ij * jnp.expand_dims(c_ij, -1)
        # (A, W) -> (1, A, W)
        x_ij = jnp.expand_dims(x_i, -3) ## Liyh: the right version
        # (1, A, W) * (A, A, W) -> (A, A, W)
        y = x_ij * w_ij
        # (A, A, W) -> (A, W)
        y = _aggregate(y, mask)
        # (A, W) -> (A, F)
        y = atomwise_ac(y, modulated_params)
        # (A, F) + (A, F) -> (A, F)
        if self.residual:
            x_new = x + y
        else:
            x_new = y

        return x_new