import jax
import jax.numpy as jnp
import pickle as pkl
import numpy as np
import os
from functools import partial 

from cybertron.readout.naive_gfn import NaiveGraphFieldNetwork, NaiveGraphFieldConditionalNetwork
from jax.sharding import PositionalSharding

from config.global_setup import EnvironConfig
SHARDING = EnvironConfig().sharding

atom_types = ['O3H2', 'N2H2', 'CarH1', 'N2H1', 'NamH0', 'Oco2H0', 'N3H3', 'N4H3', 'C1H0', 'N2H0', 'Npl3H0', 'NarH1', 'O3H1', 'C2H0', 'O3H0', 'N4H1', 'CcatH0', 'N4H2', 'CarH0', 'N3H2', 'NamH2', 'C3H1', 'Npl3H2', 'NamH1', 'C2H2', 'N3H1', 'C3H3', 'C1H1', 'C3H2', 'N1H0', 'N3H0', 'FH0', 'C2H1', 'C3H0', 'C3H4', 'NarH0', 'O2H0', 'Npl3H3', 'Npl3H1', 'O2H1']
max_num_atoms = 9

####### load trained models
net_small = NaiveGraphFieldNetwork(
    num_atom_types=len(atom_types)+1, 
    dim_atom_feature=128, 
    dim_edge_feature=128, 
    dim_atom_filter=128, 
    num_rbf_basis=128, 
    n_interactions=6, 
    cutoff=10
)

net_big = NaiveGraphFieldNetwork(
    num_atom_types=len(atom_types)+1, 
    dim_atom_feature=128, 
    dim_edge_feature=128, 
    dim_atom_filter=128, 
    num_rbf_basis=128, 
    n_interactions=6, 
    cutoff=15.0
)

net_cond_small = NaiveGraphFieldConditionalNetwork(
    num_atom_types=len(atom_types)+1, 
    num_bond_types=4,
    dim_atom_feature=128, 
    dim_bond_feature=64,
    dim_edge_feature=128, 
    dim_atom_filter=128, 
    num_rbf_basis=128, 
    n_interactions=6, 
    cutoff=10
)

net_cond_big = NaiveGraphFieldConditionalNetwork(
    num_atom_types=len(atom_types)+1, 
    num_bond_types=4,
    dim_atom_feature=128, 
    dim_bond_feature=64,
    dim_edge_feature=128, 
    dim_atom_filter=128, 
    num_rbf_basis=128, 
    n_interactions=6, 
    cutoff=15.0
)

#### unconditional networks
with open("naive_gfn_params/naive_gfn_small_jax.pkl", "rb") as f:
    params_small = pkl.load(f)
with open("naive_gfn_params/naive_gfn_big_jax.pkl", "rb") as f:
    params_big = pkl.load(f)

#### conditional networks
with open("naive_gfn_params/naive_conditional_gfn_small_jax.pkl", "rb") as f:
    params_cond_small = pkl.load(f)
with open("naive_gfn_params/naive_conditional_gfn_big_jax.pkl", "rb") as f:
    params_cond_big = pkl.load(f)

#### BGM networks 
with open("naive_gfn_params/naive_gfn_BGM_gaussian_jax.pkl", "rb") as f:
    params_bgm_gaussian = pkl.load(f)
with open("naive_gfn_params/naive_gfn_BGM_boltzmann_jax.pkl", "rb") as f:
    params_bgm_boltzmann = pkl.load(f)

if SHARDING: 
    global_sharding = PositionalSharding(jax.devices()).reshape(-1, 1)
    params_small = jax.device_put(params_small, global_sharding.replicate())
    params_big = jax.device_put(params_big, global_sharding.replicate())
    params_cond_small = jax.device_put(params_cond_small, global_sharding.replicate())
    params_cond_big = jax.device_put(params_cond_big, global_sharding.replicate())
    params_bgm_gaussian = jax.device_put(params_bgm_gaussian, global_sharding.replicate())
    params_bgm_boltzmann = jax.device_put(params_bgm_boltzmann, global_sharding.replicate())

###### jitted & vmaped functions
## unconditional networks
net_small_forward = lambda x, atom_type: net_small.apply(params_small, x, atom_type)
net_big_forward = lambda x, atom_type: net_big.apply(params_big, x, atom_type)
net_small_forward_jvj = jax.jit(
    jax.vmap(jax.jit(net_small_forward), in_axes=(0,0)))
net_big_forward_jvj = jax.jit(
    jax.vmap(jax.jit(net_big_forward), in_axes=(0,0)))

## conditional networks
net_cond_small_forward = lambda x, atom_type, bond_type: \
    net_cond_small.apply(params_cond_small, x, atom_type, bond_type) 
net_cond_big_forward = lambda x, atom_type, bond_type: \
    net_cond_big.apply(params_cond_big, x, atom_type, bond_type)
net_cond_small_forward_jvj = jax.jit(
    jax.vmap(jax.jit(net_cond_small_forward), in_axes=(0,0,0)))
net_cond_big_forward_jvj = jax.jit(
    jax.vmap(jax.jit(net_cond_big_forward), in_axes=(0,0,0)))

## BGM networks 
net_bgm_gaussian_forward = lambda x, atom_type: net_small.apply(params_bgm_gaussian, x, atom_type)
net_bgm_boltzmann_forward = lambda x, atom_type: net_small.apply(params_bgm_boltzmann, x, atom_type)
net_bgm_gaussian_forward_jvj = jax.jit(
    jax.vmap(jax.jit(net_bgm_gaussian_forward), in_axes=(0,0)))
net_bgm_boltzmann_forward_jvj = jax.jit(
    jax.vmap(jax.jit(net_bgm_boltzmann_forward), in_axes=(0,0)))    

## Langevin dynamics
def Langevin_one_step_fn(forward_fn, x, atom_type, rng_key, sigma, alpha):
    dx = forward_fn(x, atom_type)
    rng_key, normal_key = jax.random.split(rng_key)
    z = jax.random.normal(normal_key, shape=x.shape, dtype=jnp.float32)
    x = x + jnp.sqrt(2 * alpha) * z - alpha * dx / sigma
    return x, rng_key

Langevin_net_small_one_step_fn = \
    partial(Langevin_one_step_fn, net_small_forward)
Langevin_net_big_one_step_fn = \
    partial(Langevin_one_step_fn, net_big_forward)

Langevin_net_small_one_step_fn_jvj = jax.jit(
    jax.vmap(jax.jit(Langevin_net_small_one_step_fn),
             in_axes=(0,0,0,None,None)))
Langevin_net_big_one_step_fn_jvj = jax.jit(
    jax.vmap(jax.jit(Langevin_net_big_one_step_fn),
             in_axes=(0,0,0,None,None)))

def Langevin_cond_one_step_fn(forward_fn, x, atom_type, bond_type, 
                                 rng_key, sigma, alpha, gamma):
    bond_type_null = jnp.zeros_like(bond_type)
    dx = gamma * forward_fn(x, atom_type, bond_type) + \
        (1-gamma) * forward_fn(x, atom_type, bond_type_null)
    rng_key, normal_key = jax.random.split(rng_key)
    z = jax.random.normal(normal_key, shape=x.shape, dtype=jnp.float32)
    x = x + jnp.sqrt(2 * alpha) * z - alpha * dx / sigma
    return x, rng_key

Langevin_net_cond_small_one_step_fn = \
    partial(Langevin_cond_one_step_fn, net_cond_small_forward)
Langevin_net_cond_big_one_step_fn = \
    partial(Langevin_cond_one_step_fn, net_cond_big_forward)

Langevin_net_cond_small_one_step_fn_jvj = jax.jit(
    jax.vmap(jax.jit(Langevin_net_cond_small_one_step_fn),
             in_axes=(0,0,0,0,None,None,None)))
Langevin_net_cond_big_one_step_fn_jvj = jax.jit(
    jax.vmap(jax.jit(Langevin_net_cond_big_one_step_fn),
             in_axes=(0,0,0,0,None,None,None)))

def Langevin_bgm_one_step_fn(x, atom_type, rng_key,
                             sigma, alpha, t, beta_scale):
    
    dx_gauss = net_bgm_gaussian_forward(x, atom_type)
    dx_boltz = net_bgm_boltzmann_forward(x, atom_type)
    dx = dx_gauss + t * dx_boltz * beta_scale
    rng_key, normal_key = jax.random.split(rng_key)
    z = jax.random.normal(normal_key, shape=x.shape, dtype=jnp.float32)
    x = x + jnp.sqrt(2 * alpha) * z - alpha * dx / sigma
    return x, rng_key

Langevin_bgm_one_step_fn_jvj = jax.jit(
    jax.vmap(jax.jit(Langevin_bgm_one_step_fn),
             in_axes=(0,0,0,None,None,None,None)))