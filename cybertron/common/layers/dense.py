# Hyper Dense Module

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Union, Callable, Optional, Any
from flax.linen.initializers import lecun_normal, zeros_init, truncated_normal
from ..activation import get_activation
import math
from cybertron.modules.basic import safe_l2_normalize

from config.global_setup import EnvironConfig
global_setup = EnvironConfig() ### Set Hyper-parameters here
NORM_SMALL = global_setup.norm_small
BF16_FLAG = global_setup.bf16_flag
DROPOUT_FLAG = global_setup.use_dropout

Dtype = Any

class Dense(nn.Module):
    r"""A dense layer with activation function.
    
    ## Args:
        dim_out (int): Dimension of output vectors.

        activation (str, Callable): Activation function. Default: 'relu'.
    """

    dim_out: int
    use_bias: bool = True
    activation: Union[Callable, str] = "relu"
    d_type: Optional[Dtype] = jnp.float32
    kernel_init: Callable = lecun_normal()
    bias_init: Callable = zeros_init()

    @nn.compact
    def __call__(self, x):
        _dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        
        act_fn = get_activation(self.activation)
        linear_fn = nn.Dense(self.dim_out, self.use_bias, 
                             kernel_init=self.kernel_init, bias_init=self.bias_init,
                             dtype=_dtype, param_dtype=jnp.float32)

        return act_fn(linear_fn(x)) 


## Warning: this module has not been tested yet
## Warning: only support 1-dimensional ''params'' input. 
class LoRAModulatedDense(nn.Module):
    dim_out: int
    use_bias: bool = True
    activation: Union[Callable, str] = "relu"
    d_type: Optional[Dtype] = jnp.float32
    kernel_init: Callable = lecun_normal()
    bias_init: Callable = zeros_init()
    lora_rank: int = 4
    lora_alpha: Union[int, None] = None
    lora_dropout_rate: float = 0.0 
    eps: float = 1e-6
    norm_method: str = "empirical" #### "l2"    
    dtype: str = None              
    
    @nn.compact
    def __call__(self, x, params):
        # print(">>>>>input ", jnp.min(x), jnp.max(x))
        # x: (..., dim_in), params: (dim_param)
        
        _dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        _dtype if self.dtype is None else self.dtype
        
        baseline_dense = nn.Dense(
            self.dim_out, 
            self.use_bias,
            kernel_init = self.kernel_init, 
            bias_init = self.bias_init, 
            dtype = _dtype, 
            param_dtype = jnp.float32
        )
        
        x_baseline = baseline_dense(x)
        
        lora_layer_a = nn.Dense(x.shape[-1] * self.lora_rank, use_bias=False, 
                                kernel_init=lecun_normal(), name='lora_layer_a', 
                                dtype=_dtype, param_dtype=jnp.float32)
        lora_layer_b = nn.Dense(self.dim_out * self.lora_rank, use_bias=False, 
                                kernel_init=truncated_normal(), name='lora_layer_b',
                                dtype=_dtype, param_dtype=jnp.float32)#zeros_init
        #### get lora vectors
        lora_a = jnp.reshape(lora_layer_a(params), params.shape[:-1] + (self.lora_rank, x.shape[-1])) 
        lora_b = jnp.reshape(lora_layer_b(params), params.shape[:-1] + (self.lora_rank, self.dim_out))

        ##### l2 norm, control the Frobenius norm of the matrix 
        if self.norm_method == "l2":
            # lora_a = lora_a / (jnp.linalg.norm(lora_a, axis=-1, keepdims=True) + self.eps)
            # lora_b = lora_b / (jnp.linalg.norm(lora_b, axis=-1, keepdims=True) + self.eps)
            lora_a = safe_l2_normalize(lora_a, axis=-1, epsilon=NORM_SMALL)
            lora_b = safe_l2_normalize(lora_b, axis=-1, epsilon=NORM_SMALL)
        else:
            ### emperical norm 
            empirical_norm_factor_in = math.sqrt(1.0 / (x.shape[-1] * self.lora_rank))
            empirical_norm_factor_out = math.sqrt(1.0 / (self.dim_out * self.lora_rank))
            lora_a = lora_a * empirical_norm_factor_in
            lora_b = lora_b * empirical_norm_factor_out

        lora_dropout = nn.Dropout(rate=self.lora_dropout_rate, 
                                  deterministic=not DROPOUT_FLAG)
        # x = jnp.reshape(x, (x_shape[0], -1, x_shape[-1])) # (B, ?, dim_in)
        x = jnp.einsum("...ij, ...kj -> ...ik", lora_dropout(x), lora_a) # (B, ?, rank)
        # print("lora_middle: ", jnp.min(x), jnp.max(x))
        x = jnp.einsum("...ij, ...jk -> ...ik", x, lora_b)
        
        # print("x_baseline: ", jnp.min(x_baseline), jnp.max(x_baseline))
        # print("x_lora: ", jnp.min(x), jnp.max(x))
        
        ## scaling 
        if self.lora_alpha:
            lora_scaling = self.lora_alpha / self.lora_rank
        else:
            lora_scaling = 1.0
            
        x = x_baseline + lora_scaling * x
        # print(">>>>>output ", jnp.min(x), jnp.max(x))
        act_fn = get_activation(self.activation)         
        return act_fn(x)
    
    
# ## Warning: this module has not been tested yet
# ## Warning: only support 2-dimensional ''params'' input. 
# class MultiHeadLoRAModulatedDense(nn.Module):
#     dim_out: int
#     use_bias: bool = True
#     activation: Union[Callable, str] = "relu"
#     d_type: Optional[Dtype] = jnp.float32
#     kernel_init: Callable = lecun_normal()
#     bias_init: Callable = zeros_init()
#     lora_rank: int = 4
#     lora_alpha: Union[int, None] = None
#     lora_dropout_rate: float = 0.0 
#     eps: float = 1e-6
#     norm_method: str = "empirical" #### "l2"             
    
#     @nn.compact
#     def __call__(self, x, params):
        
#         baseline_dense = nn.Dense(
#             self.dim_out, 
#             self.use_bias,
#             self.d_type,
#             kernel_init = self.kernel_init, 
#             bias_init = self.bias_init
#         )            
#         x_baseline = baseline_dense(x)
        
#         vmapDense = nn.vmap(nn.Dense,
#                             in_axes=0,
#                             out_axes=0,
#                             axis_size=params.shape[0],
#                             variable_axes={'params': 0},
#                             split_rngs={'params': True, 'dropout': True})

#         lora_layer_a = vmapDense(x.shape[-1] * self.lora_rank, use_bias=False, kernel_init=lecun_normal())
#         lora_layer_b = vmapDense(self.dim_out * self.lora_rank, use_bias=False, kernel_init=truncated_normal())
        
#         #### get lora vectors
#         lora_a = jnp.reshape(lora_layer_a(params), params.shape[:-1] + (self.lora_rank, x.shape[-1])) 
#         lora_b = jnp.reshape(lora_layer_b(params), params.shape[:-1] + (self.lora_rank, self.dim_out))

#         ##### l2 norm, control the Frobenius norm of the matrix 
#         if self.norm_method == "l2":
#             lora_a = lora_a / (jnp.linalg.norm(lora_a, axis=-1, keepdims=True) + self.eps)
#             lora_b = lora_b / (jnp.linalg.norm(lora_b, axis=-1, keepdims=True) + self.eps)
#         else:
#             ### emperical norm 
#             empirical_norm_factor_in = math.sqrt(1.0 / (x.shape[-1] * self.lora_rank))
#             empirical_norm_factor_out = math.sqrt(1.0 / (self.dim_out * self.lora_rank))
#             lora_a = lora_a * empirical_norm_factor_in
#             lora_b = lora_b * empirical_norm_factor_out
        
#         x_shape = x.shape 
#         x = x.reshape(x_shape[0], -1, x_shape[-1])
#         lora_dropout = nn.Dropout(rate=self.lora_dropout_rate, 
#                                   deterministic=False)
#         # print("shape", x.shape, lora_a.shape)           
#         x = jnp.einsum("lij, lkj -> lik", lora_dropout(x), lora_a)
#         # print("shape", x.shape, lora_b.shape)
#         x = jnp.einsum("lij, ljk -> lik", x, lora_b) #### (Nhead, ?, x_out_dim)
        
#         ## scaling 
#         if self.lora_alpha:
#             lora_scaling = self.lora_alpha / self.lora_rank
#         else:
#             lora_scaling = 1.0
            
#         x = x_baseline + lora_scaling * x.reshape((-1,) + x_shape[1:-1] + (self.dim_out,))
        
#         act_fn = get_activation(self.activation)
#         return act_fn(x)

