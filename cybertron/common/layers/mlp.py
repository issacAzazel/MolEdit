"""
A mlp module from haiku.
"""

import jax
import jax.numpy as jnp
from typing import Any, Callable, Iterable, Optional, Union
from flax import linen as nn
from ..activation import get_activation
from cybertron.common.layers.dense import LoRAModulatedDense

from config.global_setup import EnvironConfig
global_setup = EnvironConfig() ### Set Hyper-parameters here
NORM_SMALL = global_setup.norm_small
BF16_FLAG = global_setup.bf16_flag
DROPOUT_FLAG = global_setup.use_dropout

Dtype = Any

class MLP(nn.Module):
    """A multi-layer perceptron module."""

    output_sizes: Iterable[int]
    w_init: Optional[nn.initializers.Initializer] = nn.initializers.xavier_uniform()
    b_init: Optional[nn.initializers.Initializer] = nn.initializers.zeros_init()
    with_bias: bool = True
    activation: Union[Callable, str] = "relu"
    activate_final: bool = False

    @nn.compact
    def __call__(self,
                 inputs: jax.Array,
                 dropout_rate: Optional[float] = None,
                 ) -> jax.Array:
        """Connects the module to some inputs.

        ## Args:
        inputs: A Tensor of shape ``[batch_size, input_size]``.

        dropout_rate: Optional dropout rate.

        rng: Optional RNG key. Require when using dropout.

        ## Returns:
        The output of the model of size ``[batch_size, output_size]``.
        """

        _dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        
        layers = []
        output_sizes = tuple(self.output_sizes)
        for index, output_size in enumerate(output_sizes):
            layers.append(nn.Dense(features=output_size,
                                   kernel_init=self.w_init,
                                   bias_init=self.b_init,
                                   use_bias=self.with_bias,
                                   dtype=_dtype,
                                   param_dtype=jnp.float32,
                                   name=f"linear_{index}"))
        layers = tuple(layers)
        output_size = output_sizes[-1] if output_sizes else None

        num_layers = len(layers)
        out = inputs
        for i, layer in enumerate(layers):
            out = layer(out)
            if i < (num_layers - 1) or self.activate_final:
                if dropout_rate is not None:
                    out = nn.Dropout(rate=dropout_rate,
                                     deterministic=not DROPOUT_FLAG)(out)
                out = get_activation(self.activation)(out)
        
        return out
    
    
class LoRAModulatedMLP(nn.Module):
    """A multi-layer perceptron module."""
    
    output_sizes: Iterable[int]
    w_init: Optional[nn.initializers.Initializer] = nn.initializers.xavier_uniform()
    b_init: Optional[nn.initializers.Initializer] = nn.initializers.zeros_init()
    with_bias: bool = True
    activation: Union[Callable, str] = "relu"
    activate_final: bool = False
    d_type: Optional[Dtype] = jnp.float32
    dropout_rate: float = 0.0
    lora_rank: int = 4
    lora_alpha: Union[int, None] = None
    lora_dropout_rate: float = 0.0 

    @nn.compact
    def __call__(self,
                 inputs: jax.Array,
                 params: jax.Array
                 ) -> jax.Array:
        """Connects the module to some inputs.

        ## Args:
        inputs: A Tensor of shape ``[batch_size, input_size]``.

        dropout_rate: Optional dropout rate.

        rng: Optional RNG key. Require when using dropout.

        ## Returns:
        The output of the model of size ``[batch_size, output_size]``.
        """

        layers = []
        output_sizes = tuple(self.output_sizes)
        for index, output_size in enumerate(output_sizes):
            layers.append(LoRAModulatedDense(
                dim_out = output_size, 
                use_bias = self.with_bias, 
                activation = self.activation, 
                d_type = self.d_type, 
                kernel_init = self.w_init,
                bias_init = self.b_init,
                lora_rank = self.lora_rank,
                lora_alpha = self.lora_alpha,
                lora_dropout_rate = self.lora_dropout_rate,
            ))
            
        layers = tuple(layers)
        output_size = output_sizes[-1] if output_sizes else None

        num_layers = len(layers)
        out = inputs
        for i, layer in enumerate(layers):
            out = layer(out, params)
            if i < (num_layers - 1) or self.activate_final:
                out = nn.Dropout(rate=self.dropout_rate,
                                 deterministic=not DROPOUT_FLAG)(out)
                out = get_activation(self.activation)(out)
        
        return out
