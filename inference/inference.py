import jax 
import jax.numpy as jnp 
from functools import partial
from tqdm import tqdm

from train.sharding import _sharding
from tqdm import tqdm
from jax.sharding import PositionalSharding

from inference.utils import Kabsch_align, permutate
from inference.fp_utils import fp_dihedral_gradient
from inference.usr_utils import USR_similarity, USR_descriptor, USR_SN_descriptor, USR_SN_similarity

def random_normals_like(rng_key, x):
    rng_key, normal_key = jax.random.split(rng_key)
    z = jax.random.normal(normal_key, shape=x.shape, dtype=jnp.float32)

    return z, rng_key

def Langevin_inference(data_dict, rng_key, score_fn, 
                       n_steps=1000, shard_inputs=True):
    epsilon = 5e-5 
    sigma_min = 0.01 # ang
    sigma_max = data_dict['rg'] / jnp.sqrt(3) # ang 
    n_eq_steps = 10
    sigmas = jnp.exp(
        jnp.linspace(jnp.log(sigma_min), jnp.log(sigma_max), n_steps)) # (nsteps, B)

    start_step = 1000
    rng_key, rnd_normal_key = jax.random.split(rng_key)
    x_t = jax.random.normal(rnd_normal_key, 
                            shape=data_dict["atom_mask"].shape + (3,), dtype=jnp.float32) * sigmas[start_step-1][..., None, None]

    input_keys = ["atom_feat", "bond_feat", "coordinates", 
                    "atom_mask", "noise_scale", "rg"]
    input_dict = {
        "atom_feat": data_dict["atom_feat"], 
        "bond_feat": data_dict["bond_feat"], 
        "coordinates": x_t ,
        "atom_mask": data_dict["atom_mask"],
        "rg": data_dict["rg"]
    }

    rng_keys = jax.random.split(rng_key, input_dict['atom_mask'].shape[0]+1)
    batch_rng_key, rng_key = rng_keys[:-1], rng_keys[-1]
    
    if shard_inputs:
        #### shard inputs 
        global_sharding = PositionalSharding(jax.devices()).reshape(len(jax.devices()), 1)
        ds_sharding = partial(_sharding, shards=global_sharding)
        input_dict = jax.tree_map(ds_sharding, input_dict)
        batch_rng_key = ds_sharding(batch_rng_key)
        
    trajectory = [x_t] 
    random_normals_like_jvj = jax.jit(jax.vmap(jax.jit(random_normals_like)))
    for step in tqdm(range(start_step-1, -1, -1)):
        sigma = sigmas[step]
        alpha = epsilon * sigma * sigma / (sigma_min * sigma_min)
        input_dict["noise_scale"] = sigma
        
        alpha = alpha[..., None, None]
        sigma = sigma[..., None, None]

        for k in range(n_eq_steps):
            dx = score_fn(*[input_dict[k] for k in input_keys])
            z, batch_rng_key = random_normals_like_jvj(batch_rng_key, x_t)
            x_t = x_t + jnp.sqrt(2 * alpha) * z - alpha * dx / sigma
            
            x_t = x_t - jnp.sum(x_t * data_dict['atom_mask'][..., None], axis=1, keepdims=True) / jnp.sum(data_dict['atom_mask'], axis=1, keepdims=True)[..., None]

            input_dict["coordinates"] = x_t
            trajectory.append(x_t)
            
    dx = score_fn(*[input_dict[k] for k in input_keys])
    x_t = x_t - sigma_min * dx
    trajectory.append(x_t)
    
    return x_t, trajectory, rng_key

def DPM_3_inference_(data_dict, rng_key, score_fn, 
                    n_steps=7, shard_inputs=True, repaint_dict=None, 
                    x_start=None, sigma_start=None, dihedral_dict=None):
    N_DPM_3 = n_steps
    n_eq_steps = 10
    
    r1 = 1.0 / 3.0
    r2 = 2.0 / 3.0
    
    epsilon = 5e-5 # 2e-4 # epsilon / (sigma_min * sigma_min) = 0.5 -> epsilon = 5e-5?
    sigma_min = 0.01 # ang
    sigma_max = data_dict['rg'] / jnp.sqrt(3) if sigma_start is None else jnp.array(sigma_start) # ang
    
    lambda_max = jnp.log(1.0 / sigma_min)
    lambda_min = jnp.log(1.0 / sigma_max)
    h = (lambda_max - lambda_min) / N_DPM_3
    
    lambda_t = lambda_min
    sigma_t = 1.0 / jnp.exp(lambda_t)

    rng_key, rnd_normal_key = jax.random.split(rng_key)
    x_t = jax.random.normal(rnd_normal_key, 
                            shape=data_dict["atom_mask"].shape + (3,), dtype=jnp.float32) * sigma_max[..., None, None]\
          if x_start is None else x_start

    input_keys = ["atom_feat", "bond_feat", "coordinates", 
                    "atom_mask", "noise_scale", "rg"]
    input_dict = {
        "atom_feat": data_dict["atom_feat"], 
        "bond_feat": data_dict["bond_feat"], 
        "coordinates": x_t ,
        "atom_mask": data_dict["atom_mask"],
        "rg": data_dict["rg"]
    }

    rng_keys = jax.random.split(rng_key, input_dict['atom_mask'].shape[0]+1)
    batch_rng_key, rng_key = rng_keys[:-1], rng_keys[-1]
    
    if shard_inputs:
        #### shard inputs 
        global_sharding = PositionalSharding(jax.devices()).reshape(len(jax.devices()), 1)
        ds_sharding = partial(_sharding, shards=global_sharding)
        input_dict = jax.tree_map(ds_sharding, input_dict)
        batch_rng_key = ds_sharding(batch_rng_key)

    trajectory = [x_t] ### debug
    random_normals_like_jvj = jax.jit(jax.vmap(jax.jit(random_normals_like)))
    if repaint_dict:
        Kabsch_align_jvj = jax.jit(jax.vmap(jax.jit(partial(Kabsch_align, is_np=False))))
        Kabsch_align_fn = Kabsch_align_jvj
        random_normals_like_jvj_repaint = jax.jit(jax.vmap(jax.jit(random_normals_like)))
        if 'permutations' in repaint_dict.keys():
            permutate_jvj = jax.jit(jax.vmap(jax.jit(permutate), in_axes=(None, 0))) ### add P-dim
            permutate_jvjvj = jax.jit(jax.vmap(permutate_jvj)) ### add B-dim
            Kabsch_align_jvj = jax.jit(jax.vmap(jax.jit(partial(Kabsch_align, is_np=False)),
                                                in_axes=(0, None, None))) ### add P-dim
            Kabsch_align_jvjvj = jax.jit(jax.vmap(Kabsch_align_jvj, in_axes=(0, 0, 0))) ### add B-dim
            Kabsch_align_fn = Kabsch_align_jvjvj
            permutate_fn = permutate_jvjvj  
            
            index_v = jax.vmap(lambda x, i: x[i])

    if dihedral_dict:
        dihedral_atom_ids = jnp.array(dihedral_dict['dihedral_atom_ids'], dtype=jnp.int32)
        fp_coeff = dihedral_dict['fp_coeff']
        fp_dihedral_gradient_fn = jax.jit(fp_dihedral_gradient)
        
    for t in tqdm(range(N_DPM_3)):        
        sigma_s1 = 1.0 / jnp.exp(lambda_t + r1 * h)
        sigma_s2 = 1.0 / jnp.exp(lambda_t + r2 * h)
        
        input_dict["noise_scale"] = sigma_t
        input_dict['coordinates'] = x_t
        dx1 = score_fn(*[input_dict[k] for k in input_keys])
        u1 = x_t - \
            (sigma_s1 * (jnp.exp(r1 * h) - 1))[..., None, None] * dx1 

        input_dict["noise_scale"] = sigma_s1
        input_dict['coordinates'] = u1
        dx2 = score_fn(*[input_dict[k] for k in input_keys])
        D1 = dx2 - dx1 

        u2 = x_t - (sigma_s2 * (jnp.exp(r2 * h) - 1))[..., None, None] * dx1\
            - (sigma_s2 * r2 / r1 * ( (jnp.exp(r2 * h) - 1) / (r2 * h) - 1))[..., None, None] * D1
        
        input_dict["noise_scale"] = sigma_s2
        input_dict['coordinates'] = u2
        dx3 = score_fn(*[input_dict[k] for k in input_keys])

        D2 = dx3 - dx1 
        lambda_t += h 
        sigma_t = 1.0 / jnp.exp(lambda_t)
        x_t = x_t - (sigma_t * (jnp.exp(h) - 1))[..., None, None] * dx1 -\
            (sigma_t / r2 * ((jnp.exp(h) - 1) / h - 1))[..., None, None] * D2

        input_dict['noise_scale'] = sigma_t
        alpha = epsilon * sigma_t * sigma_t / (sigma_min * sigma_min)
        
        alpha_ = alpha[..., None, None]
        sigma_t_ = sigma_t[..., None, None]
        for k in range(n_eq_steps):
            input_dict["coordinates"] = x_t
            dx = score_fn(*[input_dict[k] for k in input_keys])
            if dihedral_dict: ### fp trick 
                fp_grad = fp_dihedral_gradient_fn(x_t, dihedral_atom_ids)
                # print(jnp.mean(jnp.linalg.norm(fp_grad, axis=-1)[:, :26]), jnp.mean(jnp.linalg.norm(dx, axis=-1)[:, :26]))
                dx = dx + fp_coeff * fp_dihedral_gradient_fn(x_t, dihedral_atom_ids)
            z, batch_rng_key = random_normals_like_jvj(batch_rng_key, x_t)
            x_t = x_t + jnp.sqrt(2 * alpha_) * z - alpha_ * dx / sigma_t_
            if repaint_dict:
                x_cond, x_mask = repaint_dict['structure'], repaint_dict['mask']
                if 'permutations' in repaint_dict.keys():
                    x_cond = permutate_fn(x_cond, repaint_dict['permutations']) # (B, P, N, 3)
                    
                z, batch_rng_key = random_normals_like_jvj_repaint(batch_rng_key, x_cond)
                x_cond = x_cond + z * sigma_t.reshape((sigma_t.shape[0],)+(1,)*(len(z.shape)-1))
                x_cond, _, _ = Kabsch_align_fn(x_cond, x_t, x_mask)
                if 'permutations' in repaint_dict.keys():
                    MSD = jnp.sum(jnp.square(x_cond - x_t[:, None, ...]) * x_mask[:, None, :, None].astype(jnp.float32), 
                                  axis=(-1,-2))
                    msd_min_idx = jnp.argmin(MSD, axis=-1) 
                    x_cond = index_v(x_cond, msd_min_idx)
                
                trajectory.append(x_t)
                x_t = x_cond * x_mask[..., None].astype(jnp.float32) + \
                            (1.0 - x_mask[..., None].astype(jnp.float32)) * x_t
            
            x_t = x_t - jnp.sum(x_t * data_dict['atom_mask'][..., None], axis=1, keepdims=True) / jnp.sum(data_dict['atom_mask'], axis=1, keepdims=True)[..., None]

            trajectory.append(x_t)
            
    input_dict["coordinates"] = x_t
    dx = score_fn(*[input_dict[k] for k in input_keys])
    x_t = x_t - sigma_min * dx
    trajectory.append(x_t)
    
    return x_t, trajectory, rng_key


def DPM_3_inference(data_dict, rng_key, score_fn, 
                    n_steps=7, n_eq_steps=10, shard_inputs=True, repaint_dict=None, 
                    x_start=None, sigma_start=None, dihedral_dict=None, 
                    shape_dict=None):
    N_DPM_3 = n_steps
    # n_eq_steps = 10
    
    r1 = 1.0 / 3.0
    r2 = 2.0 / 3.0
    
    epsilon = 5e-5 # 2e-4 # epsilon / (sigma_min * sigma_min) = 0.5 -> epsilon = 5e-5?
    sigma_min = 0.01 # ang
    sigma_max = data_dict['rg'] / jnp.sqrt(3) if sigma_start is None else jnp.array(sigma_start) # ang
    
    lambda_max = jnp.log(1.0 / sigma_min)
    lambda_min = jnp.log(1.0 / sigma_max)
    h = (lambda_max - lambda_min) / N_DPM_3
    
    lambda_t = lambda_min
    sigma_t = 1.0 / jnp.exp(lambda_t)

    rng_key, rnd_normal_key = jax.random.split(rng_key)
    x_t = jax.random.normal(rnd_normal_key, 
                            shape=data_dict["atom_mask"].shape + (3,), dtype=jnp.float32) * sigma_max[..., None, None]\
          if x_start is None else x_start

    input_keys = ["atom_feat", "bond_feat", "coordinates", 
                    "atom_mask", "noise_scale", "rg"]
    if "property" in data_dict: input_keys += ['property']
    input_dict = {
        "atom_feat": data_dict["atom_feat"], 
        "bond_feat": data_dict["bond_feat"], 
        "coordinates": x_t ,
        "atom_mask": data_dict["atom_mask"],
        "rg": data_dict["rg"]
    }
    if "property" in data_dict: input_dict.update({"property": data_dict["property"]})

    rng_keys = jax.random.split(rng_key, input_dict['atom_mask'].shape[0]+1)
    batch_rng_key, rng_key = rng_keys[:-1], rng_keys[-1]
    
    if shard_inputs:
        #### shard inputs 
        global_sharding = PositionalSharding(jax.devices()).reshape(len(jax.devices()), 1)
        ds_sharding = partial(_sharding, shards=global_sharding)
        input_dict = jax.tree_map(ds_sharding, input_dict)
        batch_rng_key = ds_sharding(batch_rng_key)

        # #### shard repaint_dict & shape dict ?? ### CHECK this! 20240908
        # if repaint_dict: repaint_dict = jax.tree_map(ds_sharding, repaint_dict)
        # if shape_dict: shape_dict = jax.tree_map(ds_sharding, shape_dict)

    trajectory = [x_t] ### debug
    random_normals_like_jvj = jax.jit(jax.vmap(jax.jit(random_normals_like)))
    if repaint_dict:
        Kabsch_align_jvj = jax.jit(jax.vmap(jax.jit(partial(Kabsch_align, is_np=False))))
        Kabsch_align_fn = Kabsch_align_jvj
        random_normals_like_jvj_repaint = jax.jit(jax.vmap(jax.jit(random_normals_like)))
        if 'permutations' in repaint_dict.keys():
            permutate_jvj = jax.jit(jax.vmap(jax.jit(permutate), in_axes=(None, 0))) ### add P-dim
            permutate_jvjvj = jax.jit(jax.vmap(permutate_jvj)) ### add B-dim
            Kabsch_align_jvj = jax.jit(jax.vmap(jax.jit(partial(Kabsch_align, is_np=False)),
                                                in_axes=(0, None, None))) ### add P-dim
            Kabsch_align_jvjvj = jax.jit(jax.vmap(Kabsch_align_jvj, in_axes=(0, 0, 0))) ### add B-dim
            Kabsch_align_fn = Kabsch_align_jvjvj
            permutate_fn = permutate_jvjvj  
            
            index_v = jax.vmap(lambda x, i: x[i])

    if dihedral_dict:
        dihedral_atom_ids = jnp.array(dihedral_dict['dihedral_atom_ids'], dtype=jnp.int32)
        fp_coeff = dihedral_dict['fp_coeff']
        fp_dihedral_gradient_fn = jax.jit(fp_dihedral_gradient)

    if shape_dict:
        template_x = jnp.array(shape_dict['template_structure'], dtype=jnp.float32)
        template_atom_mask = jnp.array(shape_dict['template_atom_mask'], dtype=jnp.float32)
        shape_template_coeff = shape_dict['template_coeff']
        
        usr_similarity_value_and_grad_fn = jax.jit(jax.vmap(jax.jit(jax.value_and_grad(USR_similarity))))
        
    for t in tqdm(range(N_DPM_3)):        
        sigma_s1 = 1.0 / jnp.exp(lambda_t + r1 * h)
        sigma_s2 = 1.0 / jnp.exp(lambda_t + r2 * h)
        
        input_dict["noise_scale"] = sigma_t
        input_dict['coordinates'] = x_t
        dx1 = score_fn(*[input_dict[k] for k in input_keys])
        u1 = x_t - \
            (sigma_s1 * (jnp.exp(r1 * h) - 1))[..., None, None] * dx1 

        input_dict["noise_scale"] = sigma_s1
        input_dict['coordinates'] = u1
        dx2 = score_fn(*[input_dict[k] for k in input_keys])
        D1 = dx2 - dx1 

        u2 = x_t - (sigma_s2 * (jnp.exp(r2 * h) - 1))[..., None, None] * dx1\
            - (sigma_s2 * r2 / r1 * ( (jnp.exp(r2 * h) - 1) / (r2 * h) - 1))[..., None, None] * D1
        
        input_dict["noise_scale"] = sigma_s2
        input_dict['coordinates'] = u2
        dx3 = score_fn(*[input_dict[k] for k in input_keys])

        D2 = dx3 - dx1 
        lambda_t += h 
        sigma_t = 1.0 / jnp.exp(lambda_t)
        x_t = x_t - (sigma_t * (jnp.exp(h) - 1))[..., None, None] * dx1 -\
           (sigma_t / r2 * ((jnp.exp(h) - 1) / h - 1))[..., None, None] * D2

        input_dict['noise_scale'] = sigma_t
        alpha = epsilon * sigma_t * sigma_t / (sigma_min * sigma_min)
        
        alpha_ = alpha[..., None, None]
        sigma_t_ = sigma_t[..., None, None]
        for k in range(n_eq_steps):
            input_dict["coordinates"] = x_t
            dx = score_fn(*[input_dict[k] for k in input_keys])
            if dihedral_dict: ### fp trick 
                fp_grad = fp_dihedral_gradient_fn(x_t, dihedral_atom_ids)
                # print(jnp.mean(jnp.linalg.norm(fp_grad, axis=-1)[:, :26]), jnp.mean(jnp.linalg.norm(dx, axis=-1)[:, :26]))
                dx = dx + fp_coeff * fp_dihedral_gradient_fn(x_t, dihedral_atom_ids)

            if shape_dict: #### shape guidance 
                usr_sim, usr_grad = usr_similarity_value_and_grad_fn(x_t, template_x, input_dict['atom_mask'], template_atom_mask)
                # print(usr_sim[0], jnp.mean(jnp.linalg.norm(usr_grad, axis=-1)[:, :26]), jnp.mean(jnp.linalg.norm(dx, axis=-1)[:, :26]))
                # dx = dx - shape_template_coeff * jnp.exp(- 4.0 * t / N_DPM_3) * usr_grad
                dx = dx - shape_template_coeff * usr_grad
            
            z, batch_rng_key = random_normals_like_jvj(batch_rng_key, x_t)
            x_t = x_t + jnp.sqrt(2 * alpha_) * z - alpha_ * dx / sigma_t_
            
            if repaint_dict: #### repaint trick
                x_cond, x_mask = repaint_dict['structure'], repaint_dict['mask']
                if 'permutations' in repaint_dict.keys():
                    x_cond = permutate_fn(x_cond, repaint_dict['permutations']) # (B, P, N, 3)
                    
                z, batch_rng_key = random_normals_like_jvj_repaint(batch_rng_key, x_cond)
                x_cond = x_cond + z * sigma_t.reshape((sigma_t.shape[0],)+(1,)*(len(z.shape)-1))
                x_cond, _, _ = Kabsch_align_fn(x_cond, x_t, x_mask)
                if 'permutations' in repaint_dict.keys():
                    MSD = jnp.sum(jnp.square(x_cond - x_t[:, None, ...]) * x_mask[:, None, :, None].astype(jnp.float32), 
                                  axis=(-1,-2))
                    msd_min_idx = jnp.argmin(MSD, axis=-1) 
                    x_cond = index_v(x_cond, msd_min_idx)
                
                trajectory.append(x_t)
                x_t = x_cond * x_mask[..., None].astype(jnp.float32) + \
                            (1.0 - x_mask[..., None].astype(jnp.float32)) * x_t
            
            x_t = x_t - jnp.sum(x_t * data_dict['atom_mask'][..., None], axis=1, keepdims=True) / jnp.sum(data_dict['atom_mask'], axis=1, keepdims=True)[..., None]

            trajectory.append(x_t)
            
    input_dict["coordinates"] = x_t
    dx = score_fn(*[input_dict[k] for k in input_keys])
    x_t = x_t - sigma_min * dx
    trajectory.append(x_t)
    
    return x_t, trajectory, rng_key

def DPM_3_inference_modified_shape_guidance(data_dict, rng_key, score_fn, 
                                            n_steps=7, shard_inputs=True, repaint_dict=None, 
                                            x_start=None, sigma_start=None, dihedral_dict=None, 
                                            shape_dict=None):
    N_DPM_3 = n_steps
    n_eq_steps = 10
    
    r1 = 1.0 / 3.0
    r2 = 2.0 / 3.0
    
    epsilon = 5e-5 # 2e-4 # epsilon / (sigma_min * sigma_min) = 0.5 -> epsilon = 5e-5?
    sigma_min = 0.01 # ang
    sigma_max = data_dict['rg'] / jnp.sqrt(3) if sigma_start is None else jnp.array(sigma_start) # ang
    
    lambda_max = jnp.log(1.0 / sigma_min)
    lambda_min = jnp.log(1.0 / sigma_max)
    h = (lambda_max - lambda_min) / N_DPM_3
    
    lambda_t = lambda_min
    sigma_t = 1.0 / jnp.exp(lambda_t)

    rng_key, rnd_normal_key = jax.random.split(rng_key)
    x_t = jax.random.normal(rnd_normal_key, 
                            shape=data_dict["atom_mask"].shape + (3,), dtype=jnp.float32) * sigma_max[..., None, None]\
          if x_start is None else x_start

    input_keys = ["atom_feat", "bond_feat", "coordinates", 
                    "atom_mask", "noise_scale", "rg"]
    input_dict = {
        "atom_feat": data_dict["atom_feat"], 
        "bond_feat": data_dict["bond_feat"], 
        "coordinates": x_t ,
        "atom_mask": data_dict["atom_mask"],
        "gaussian_std": data_dict["gaussian_std"] if "gaussian_std" in data_dict else jnp.zeros_like(data_dict["atom_mask"], dtype=jnp.float32),
        "rg": data_dict["rg"]
    }

    rng_keys = jax.random.split(rng_key, input_dict['atom_mask'].shape[0]+1)
    batch_rng_key, rng_key = rng_keys[:-1], rng_keys[-1]
    
    if shard_inputs:
        #### shard inputs 
        global_sharding = PositionalSharding(jax.devices()).reshape(len(jax.devices()), 1)
        ds_sharding = partial(_sharding, shards=global_sharding)
        input_dict = jax.tree_map(ds_sharding, input_dict)
        batch_rng_key = ds_sharding(batch_rng_key)

        # #### shard repaint_dict & shape dict ?? ### CHECK this! 20240908
        # if repaint_dict: repaint_dict = jax.tree_map(ds_sharding, repaint_dict)
        # if shape_dict: shape_dict = jax.tree_map(ds_sharding, shape_dict)

    trajectory = [x_t] ### debug
    random_normals_like_jvj = jax.jit(jax.vmap(jax.jit(random_normals_like)))
    if repaint_dict:
        Kabsch_align_jvj = jax.jit(jax.vmap(jax.jit(partial(Kabsch_align, is_np=False))))
        Kabsch_align_fn = Kabsch_align_jvj
        random_normals_like_jvj_repaint = jax.jit(jax.vmap(jax.jit(random_normals_like)))
        if 'permutations' in repaint_dict.keys():
            permutate_jvj = jax.jit(jax.vmap(jax.jit(permutate), in_axes=(None, 0))) ### add P-dim
            permutate_jvjvj = jax.jit(jax.vmap(permutate_jvj)) ### add B-dim
            Kabsch_align_jvj = jax.jit(jax.vmap(jax.jit(partial(Kabsch_align, is_np=False)),
                                                in_axes=(0, None, None))) ### add P-dim
            Kabsch_align_jvjvj = jax.jit(jax.vmap(Kabsch_align_jvj, in_axes=(0, 0, 0))) ### add B-dim
            Kabsch_align_fn = Kabsch_align_jvjvj
            permutate_fn = permutate_jvjvj  
            
            index_v = jax.vmap(lambda x, i: x[i])

    if dihedral_dict:
        dihedral_atom_ids = jnp.array(dihedral_dict['dihedral_atom_ids'], dtype=jnp.int32)
        fp_coeff = dihedral_dict['fp_coeff']
        fp_dihedral_gradient_fn = jax.jit(fp_dihedral_gradient)

    if shape_dict:
        template_x = jnp.array(shape_dict['template_structure'], dtype=jnp.float32)
        template_std = jnp.array(shape_dict['template_std'], dtype=jnp.float32)
        template_atom_mask = jnp.array(shape_dict['template_atom_mask'], dtype=jnp.float32)
        template_key_group_ids = jnp.array(shape_dict['template_key_group_ids'], dtype=jnp.int32)
        mol_key_group_ids = jnp.array(shape_dict['mol_key_group_ids'], dtype=jnp.int32)
        shape_template_coeff = shape_dict['template_coeff']
        gaussian_scale_factor = shape_dict['gaussian_scale_factor']

        usr_similarity_value_and_grad_fn = jax.jit(jax.vmap(jax.jit(jax.value_and_grad(USR_similarity))))
        usr_sn_similarity_value_and_grad_fn = jax.jit(jax.vmap(jax.jit(jax.value_and_grad(USR_SN_similarity, has_aux=True))))
        
    for t in tqdm(range(N_DPM_3)):        
        sigma_s1 = 1.0 / jnp.exp(lambda_t + r1 * h)
        sigma_s2 = 1.0 / jnp.exp(lambda_t + r2 * h)
        
        input_dict["noise_scale"] = sigma_t
        input_dict['coordinates'] = x_t
        dx1 = score_fn(*[input_dict[k] for k in input_keys])
        u1 = x_t - \
            (sigma_s1 * (jnp.exp(r1 * h) - 1))[..., None, None] * dx1 

        input_dict["noise_scale"] = sigma_s1
        input_dict['coordinates'] = u1
        dx2 = score_fn(*[input_dict[k] for k in input_keys])
        D1 = dx2 - dx1 

        u2 = x_t - (sigma_s2 * (jnp.exp(r2 * h) - 1))[..., None, None] * dx1\
            - (sigma_s2 * r2 / r1 * ( (jnp.exp(r2 * h) - 1) / (r2 * h) - 1))[..., None, None] * D1
        
        input_dict["noise_scale"] = sigma_s2
        input_dict['coordinates'] = u2
        dx3 = score_fn(*[input_dict[k] for k in input_keys])

        D2 = dx3 - dx1 
        lambda_t += h 
        sigma_t = 1.0 / jnp.exp(lambda_t)
        x_t = x_t - (sigma_t * (jnp.exp(h) - 1))[..., None, None] * dx1 -\
           (sigma_t / r2 * ((jnp.exp(h) - 1) / h - 1))[..., None, None] * D2

        input_dict['noise_scale'] = sigma_t
        alpha = epsilon * sigma_t * sigma_t / (sigma_min * sigma_min)
        
        alpha_ = alpha[..., None, None]
        sigma_t_ = sigma_t[..., None, None]
        for k in range(n_eq_steps):
            input_dict["coordinates"] = x_t
            dx = score_fn(*[input_dict[k] for k in input_keys])
            if dihedral_dict: ### fp trick 
                fp_grad = fp_dihedral_gradient_fn(x_t, dihedral_atom_ids)
                # print(jnp.mean(jnp.linalg.norm(fp_grad, axis=-1)[:, :26]), jnp.mean(jnp.linalg.norm(dx, axis=-1)[:, :26]))
                dx = dx + fp_coeff * fp_dihedral_gradient_fn(x_t, dihedral_atom_ids)

            if shape_dict: #### shape guidance 
                # usr_sim, usr_grad = usr_similarity_value_and_grad_fn(x_t, template_x, input_dict['atom_mask'], template_atom_mask)
                # USR_SN_similarity(structure_1, structure_2, stds_1, stds_2, atom_mask_1, atom_mask_2, rng_key):
                (usr_sn_sim, batch_rng_key), usr_sn_grad = \
                usr_sn_similarity_value_and_grad_fn(x_t, template_x, data_dict['gaussian_std'] * gaussian_scale_factor,
                                                    template_std * gaussian_scale_factor, input_dict['atom_mask'], template_atom_mask, 
                                                    batch_rng_key, mol_key_group_ids, template_key_group_ids)
                # print(usr_sim[0], jnp.mean(jnp.linalg.norm(usr_grad, axis=-1)[:, :26]), jnp.mean(jnp.linalg.norm(dx, axis=-1)[:, :26]))
                # dx = dx - shape_template_coeff * jnp.exp(- 4.0 * t / N_DPM_3) * usr_grad
                # dx = dx - shape_template_coeff * usr_grad
                dx = dx - shape_template_coeff * usr_sn_grad
            
            z, batch_rng_key = random_normals_like_jvj(batch_rng_key, x_t)
            x_t = x_t + jnp.sqrt(2 * alpha_) * z - alpha_ * dx / sigma_t_
            
            if repaint_dict: #### repaint trick
                x_cond, x_mask = repaint_dict['structure'], repaint_dict['mask']
                if 'permutations' in repaint_dict.keys():
                    x_cond = permutate_fn(x_cond, repaint_dict['permutations']) # (B, P, N, 3)
                    
                z, batch_rng_key = random_normals_like_jvj_repaint(batch_rng_key, x_cond)
                x_cond = x_cond + z * sigma_t.reshape((sigma_t.shape[0],)+(1,)*(len(z.shape)-1))
                x_cond, _, _ = Kabsch_align_fn(x_cond, x_t, x_mask)
                if 'permutations' in repaint_dict.keys():
                    MSD = jnp.sum(jnp.square(x_cond - x_t[:, None, ...]) * x_mask[:, None, :, None].astype(jnp.float32), 
                                  axis=(-1,-2))
                    msd_min_idx = jnp.argmin(MSD, axis=-1) 
                    x_cond = index_v(x_cond, msd_min_idx)
                
                trajectory.append(x_t)
                x_t = x_cond * x_mask[..., None].astype(jnp.float32) + \
                            (1.0 - x_mask[..., None].astype(jnp.float32)) * x_t
            
            x_t = x_t - jnp.sum(x_t * data_dict['atom_mask'][..., None], axis=1, keepdims=True) / jnp.sum(data_dict['atom_mask'], axis=1, keepdims=True)[..., None]

            trajectory.append(x_t)
            
    input_dict["coordinates"] = x_t
    dx = score_fn(*[input_dict[k] for k in input_keys])
    x_t = x_t - sigma_min * dx
    trajectory.append(x_t)
    
    return x_t, trajectory, rng_key

def DPM_pp_2S_inference(data_dict, rng_key, score_fn, 
                        n_steps=10, shard_inputs=True):
    N_DPM_2 = n_steps
    n_eq_steps = 10
    epsilon = 5e-5 # 2e-4 # epsilon / (sigma_min * sigma_min) = 0.5 -> epsilon = 5e-5?
    sigma_min = 0.01 # ang
    sigma_max = data_dict['rg'] / jnp.sqrt(3) # ang
    
    lambda_max = jnp.log(1.0 / sigma_min)
    lambda_min = jnp.log(1.0 / sigma_max)
    h = (lambda_max - lambda_min) / N_DPM_2
    
    lambda_t = lambda_min
    sigma_t = 1.0 / jnp.exp(lambda_t)

    rng_key, rnd_normal_key = jax.random.split(rng_key)
    x_t = jax.random.normal(rnd_normal_key, 
                            shape=data_dict["atom_mask"].shape + (3,), dtype=jnp.float32) * sigma_max[..., None, None]

    input_keys = ["atom_feat", "bond_feat", "coordinates", 
                    "atom_mask", "noise_scale", "rg"]
    input_dict = {
        "atom_feat": data_dict["atom_feat"], 
        "bond_feat": data_dict["bond_feat"], 
        "coordinates": x_t ,
        "atom_mask": data_dict["atom_mask"],
        "noise_scale": data_dict["noise_scale"],
        "rg": data_dict["rg"]
    }

    rng_keys = jax.random.split(rng_key, input_dict['atom_mask'].shape[0]+1)
    batch_rng_key, rng_key = rng_keys[:-1], rng_keys[-1]
    
    if shard_inputs:
        #### shard inputs 
        global_sharding = PositionalSharding(jax.devices()).reshape(len(jax.devices()), 1)
        ds_sharding = partial(_sharding, shards=global_sharding)
        input_dict = jax.tree_map(ds_sharding, input_dict)
        batch_rng_key = ds_sharding(batch_rng_key)

    trajectory = [x_t] ### debug
    random_normals_like_jvj = jax.jit(jax.vmap(jax.jit(random_normals_like)))
    
    for t in tqdm(range(N_DPM_2)):
        rt = 1.0 / 2.0
        input_dict['coordinates'] = x_t
        input_dict['noise_scale'] = sigma_t
        dx1 = score_fn(*[input_dict[k] for k in input_keys])

        x_theta = x_t - sigma_t[..., None, None] * dx1
        st = lambda_t + rt * h 
        sigma_st = 1.0 / jnp.exp(st)
        sigma_t1 = 1.0 / jnp.exp(lambda_t + h)
        ui = (sigma_st / sigma_t)[..., None, None] *  x_t - (jnp.exp(-rt * h) - 1)[..., None, None] * x_theta
        
        x_theta2 = x_theta
        
        input_dict['coordinates'] = ui
        input_dict['noise_scale'] = sigma_st
        dx3 = score_fn(*[input_dict[k] for k in input_keys])
        Di = (1.0 - 1.0/(2.0 * rt)) * x_theta2 + 1.0 / (2 * rt) * (x_t - sigma_st[..., None, None] * dx3)
        x_t = (sigma_t1 / sigma_t)[..., None, None] * x_t - (jnp.exp(- h) - 1)[..., None, None] * Di

        lambda_t += h 
        sigma_t = sigma_t1
        
        input_dict['noise_scale'] = sigma_t
        alpha = epsilon * sigma_t * sigma_t / (sigma_min * sigma_min)
        for k in range(n_eq_steps):
            input_dict["coordinates"] = x_t
            dx = score_fn(*[input_dict[k] for k in input_keys])
            z, batch_rng_key = random_normals_like_jvj(batch_rng_key, x_t)
            x_t = x_t + jnp.sqrt(2 * alpha)[..., None, None] * z - alpha[..., None, None] * dx / sigma_t[..., None, None]
            
            x_t = x_t - jnp.sum(x_t * data_dict['atom_mask'][..., None], axis=1, keepdims=True) / jnp.sum(data_dict['atom_mask'], axis=1, keepdims=True)[..., None]

            trajectory.append(x_t)

    dx = score_fn(*[input_dict[k] for k in input_keys])
    x_t = x_t - sigma_min * dx 
    trajectory.append(x_t)
    