"""
A mlp module from haiku.
"""

import jax
import jax.numpy as jnp
from typing import Any, Callable, Iterable, Optional, Union
from flax import linen as nn
from ..activation import get_activation

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

        layers = []
        output_sizes = tuple(self.output_sizes)
        for index, output_size in enumerate(output_sizes):
            layers.append(nn.Dense(features=output_size,
                                   kernel_init=self.w_init,
                                   bias_init=self.b_init,
                                   use_bias=self.with_bias,
                                   name=f"linear_{index}"))
        layers = tuple(layers)
        output_size = output_sizes[-1] if output_sizes else None

        num_layers = len(layers)
        out = inputs
        for i, layer in enumerate(layers):
            out = layer(out)
            if i < (num_layers - 1) or self.activate_final:
                if dropout_rate is not None:
                    out = nn.Dropout(rate=dropout_rate)(out)
                out = get_activation(self.activation)(out)
        
        return out