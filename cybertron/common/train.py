# Code for training a cybertron model.
# 23-11-02

import jax
import optax
import jax.numpy as jnp
import numpy as np

from flax.training import common_utils
from typing import Union, Tuple, List, Optional
from jax import nn as nn
from jax.numpy import ndarray as ndarray
from optax import Schedule

Array = Union[jax.Array, np.ndarray]

def print_net_params(params: dict):
    """Print all params with shape like a tree."""
    count = 0

    def _print_net_params(params: dict, ret: int = 0,):

        for k, v in params.items():
            if isinstance(v, dict):
                print(" "*ret + f"{k}:")
                _print_net_params(v, ret+2)
            else:
                print(" " * (ret+2) + f"{k}: {v.shape}")

    _print_net_params(params)
    param_arrays = jax.tree_util.tree_leaves(params)
    for p in param_arrays:
        count += p.size
    print(f"Total number of parameters: {count}")

## Pmap training utils
def shard_array(array: Array) -> jax.Array:
  """Shards `array` along its leading dimension."""
  return jax.device_put_sharded(
      shards=list(common_utils.shard(array)),
      devices=jax.devices())

## Learning schedule
def polynomial_decay_schedule(init_value: float,
                              power: float,
                              transition_begin: int = 0,) -> Schedule:
    
    def schedule(count):
        count += transition_begin
        return init_value * jnp.power(count, power)

    return schedule
    
def transformer_lr(learning_rate: float = 1.0,
                   warmup_steps: int = 4000,
                   dimension: int = 1,):
    
    dim_scale = np.power(dimension, -0.5)
    max_lr = learning_rate * dim_scale * np.power(warmup_steps, -0.5)
    print(f"[Transformer LR] max_lr: {max_lr:.4e}")

    warmup_fn = optax.linear_schedule(init_value=0.0,
                                      end_value=max_lr,
                                      transition_steps=warmup_steps)
    decay_fn = polynomial_decay_schedule(init_value=learning_rate * dim_scale,
                                         power=-0.5,
                                         transition_begin=warmup_steps,)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn],
                                       boundaries=[warmup_steps])
    
    return schedule_fn
                                         

## data utils
def batch_split(data: Array, batch_size: int, drop_remainder: bool = False):
    r"""Split one array into a list of arrays with batch size."""

    if data is None:
        return None

    if drop_remainder:
        num_batch = data.shape[0] // batch_size
        batches = np.array_split(data[:num_batch * batch_size], num_batch)
        return batches
    else:
        data_size = data.shape[0]
        num_batch = (data_size + batch_size - 1) // batch_size
        batches = []

        for i in range(num_batch):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, data_size)
            batches.append(data[start_idx:end_idx])
        
        return batches

def create_batches(data_dict: dict, 
                   batch_size: int, 
                   data_size: int, 
                   input_keys: Union[list, tuple]):

    num_batch = (data_size + batch_size - 1) // batch_size
    print("[Batch spliter] Create batch number:", num_batch)

    batches = []
    for k, v in data_dict.items():
        if v is None:
            batches.append([None for _ in range(num_batch)])
        elif k in input_keys:
            batches.append(batch_split(v, batch_size, drop_remainder=False))
        else:
            pass
    
    out = []
    for i in range(num_batch):
        batch = {}
        for j, k in enumerate(input_keys):
            batch[k] = batches[j][i]
        out.append(batch)

    return out

## Scale shift methods
class ScaleShift:

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        
    def calc(self, x: ndarray, **kwargs):
        raise NotImplementedError

class LinearScaleShift(ScaleShift):

    def __init__(self, 
                 a: float = 1.0, 
                 b: float = 0.0, 
                 **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def calc(self, x: ndarray):
        return x * self.a + self.b