import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from flax.linen.initializers import zeros_init, xavier_uniform
from typing import Any, Optional, Union, Callable

from .activation import get_activation
from config.global_setup import EnvironConfig
global_setup = EnvironConfig() ### Set Hyper-parameters here
NORM_SMALL = global_setup.norm_small
BF16_FLAG = global_setup.bf16_flag
DROPOUT_FLAG = global_setup.use_dropout
REMAT_FLAG = global_setup.remat_flag

class NormBlock(nn.Module):

    norm_method: str = "layernorm"
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):

        if self.norm_method == "layernorm":
            x_safe = jnp.float32(x)
            x_safe = nn.LayerNorm(epsilon=self.eps)(x_safe)
            return x_safe.astype(x.dtype)
        elif self.norm_method == "rmsnorm":
            x_safe = jnp.float32(x)
            x_safe = nn.RMSNorm(epsilon=self.eps)(x_safe)
            return x_safe.astype(x.dtype)
        else:
            raise ValueError(f"Unsupported norm method: {self.norm_method}")

class adaLN(nn.Module):

    module: nn.Module
    activation: str = 'silu'

    @nn.compact
    def __call__(self, x, cond, other_inputs = ()):
        #### Input: x: (..., F), cond: (F)

        #### 1. generate alpha, gamma, beta
        hidden_size = x.shape[-1]
        arr_dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        cond = get_activation(self.activation)(cond)
        cond = nn.Dense(
            features = 3 * hidden_size,
            kernel_init = xavier_uniform(),
            dtype = arr_dtype,
            param_dtype = jnp.float32,
        )(cond) # (3 * F)
        alpha, beta, gamma = jnp.split(cond, 3, -1) # (F)
        alpha, beta, gamma = jax.tree_util.tree_map(
            lambda val: jnp.reshape(val, (1,) * (x.ndim - 1) + (hidden_size,)),
            (alpha, beta, gamma),
        )

        #### 2. main function
        norm_small = NORM_SMALL
        act, d_act = x, x
        d_act = NormBlock(eps = norm_small)(d_act)
        d_act = d_act * (1 + gamma) + beta
        d_act = self.module(d_act, *other_inputs)
        d_act = d_act * alpha
        act += d_act

        return act