"""Model config."""

import copy
import ml_collections
import jax 
import jax.numpy as jnp

def EnvironConfig() -> ml_collections.ConfigDict:
    """Get the ConfigDict."""
    cfg = copy.deepcopy(CONFIG)
    return cfg

CONFIG = ml_collections.ConfigDict({
    'int_dtype': jnp.int32,
    'float_dtype': jnp.float32, 
    'epsilon': 1e-6,
    'sharding': True, 
    'use_dropout': True,
    'global_dropout_rate': None,
    'remat_flag': False, 
    'safe_precision_flag': True, #False,
    'bf16_flag': True, #False,
    'norm_small': 1e-5
})
