from cybertron.common.config_load import Config
import jax 
import jax.numpy as jnp
from flax import traverse_util
import numpy as np 
from functools import reduce

def logger(f, logger_info, flush=False):
    f.write(logger_info + "\n")
    print(logger_info)
    if (flush):
        f.flush()

def loss_logger(f, loss_dict, prefix=""):
    for k, v in loss_dict.items():
        if isinstance(v, dict):
            logger(f, "{}{}:".format(prefix, k))
            loss_logger(f, v, prefix=prefix + "\t")
        else:
            logger(f, "{}{}: {:.4f}".format(prefix, k, v))

def split_multiple_rng_keys(rng_key, num_keys):
    rng_keys = jax.random.split(rng_key, num_keys + 1)
    return rng_keys[:-1], rng_keys[-1]

def set_dropout_rate_config(d, dropout_rate):
    if isinstance(d, Config):
        d = d.__dict__
    for k, v in d.items():
        if isinstance(v, dict) or isinstance(v, Config):
            d[k] = set_dropout_rate_config(v, dropout_rate)
        else:
            d[k] = dropout_rate if "dropout" in k else v    
    return Config(d)

def parameter_weight_decay(params):
    """Apply weight decay to parameters."""
    
    # loss = jax.tree_util.tree_map(
    #     lambda p: jnp.mean(
    #             jnp.square(p.reshape(-1))
    #         ) if p.ndim == 2 else 0, params)
    loss = traverse_util.path_aware_map(
        lambda p, x: jnp.mean(
                jnp.square(x.reshape(-1))
            ) if 'kernel' in p else 0, params)
    loss = jnp.sum(
        jnp.array(jax.tree_util.tree_leaves(loss))
    )
    
    return loss

def any_nan_in_tree(tree):
    
    is_nan_tree = jax.tree_util.tree_map(
        lambda p: jnp.sum(jnp.isnan(p).astype(jnp.float32)),
        tree 
    )
    
    return jnp.sum(
        jnp.array(jax.tree_util.tree_leaves(is_nan_tree))) > 0.0
    
# def orgnize_name_list(train_name_list_bin, bins, batch_size):
#     max_bin_size = np.max([train_name_list_bin[b]['size'] for b in bins])
#     max_batches = max_bin_size // batch_size + 1

#     for b in bins:
#         np.random.shuffle(train_name_list_bin[b]['name_list'])
#         train_name_list_bin[b]['name_list'] = train_name_list_bin[b]['name_list'] * 2

#     orgnized_name_list = []

#     for batch in range(max_batches):
#         for b in bins:
#             orgnized_name_list.extend(train_name_list_bin[b]['name_list'][batch*batch_size: (batch + 1) * batch_size])
            
#     return orgnized_name_list


def orgnize_name_list(train_name_list_bin, bins, batch_size, num_batches, p_scaling=1.0):
    bin_prob = \
        np.array([train_name_list_bin[b]['size'] for b in bins], dtype=np.float32) * p_scaling
    select_bin_idx = \
        np.random.choice(np.arange(len(bins)), 
                         size=(num_batches,), 
                         p = bin_prob/np.sum(bin_prob))
    select_bins = [bins[i] for i in select_bin_idx]
    
    # name_list = [
    #     np.random.choice(train_name_list_bin[b]['name_list'],
    #                      size=(batch_size,)) for b in select_bins
    # ]
    orgnized_name_list = [] # reduce(lambda x, y: x+y, name_list)
    for b in select_bins:
        select_file_idx = np.random.randint(0, 
                                            train_name_list_bin[b]['size'], 
                                            size=(batch_size,))
        orgnized_name_list.extend([train_name_list_bin[b]['name_list'][i] \
                                    for i in select_file_idx])

    return orgnized_name_list, select_bins