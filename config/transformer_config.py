"""Example configuration file for protoken generator.
"""

import jax
import jax.numpy as jnp
from ml_collections.config_dict import ConfigDict


transformer_config = {
    "vocab_size": None,
    "output_vocab_size": None, 
    "share_embeddings": False,
    "logits_via_embedding": False,
    "dtype": jnp.bfloat16,
    "emb_dim": 1024,
    "num_heads": 16,
    "num_layers": 6,
    "qkv_dim": 1024,
    "mlp_dim": 4096,
    "max_len": 512,
    "dropout_rate": 0.1,
    "attention_dropout_rate": 0.1,
    "deterministic": False,
}

transformer_config = ConfigDict(transformer_config)