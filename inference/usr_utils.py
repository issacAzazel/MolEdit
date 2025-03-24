import jax
import jax.numpy as jnp
from functools import partial

#### USR descriptor from paper: Ultrafast shape recognition to search compound databases for smiliar molecular shapes

def USR_descriptor(x, atom_mask):
    x, atom_mask = jnp.array(x).astype(jnp.float32), jnp.array(atom_mask).astype(jnp.float32)
    ctd = jnp.sum(x * atom_mask[..., None], axis=0) / jnp.sum(atom_mask)
    distance_to_ctd = jnp.linalg.norm(x - ctd[None, ...] + (1.0 - atom_mask[..., None]) * 1e-6, axis=-1)
    moment_ctd_1 = jnp.sum(distance_to_ctd * atom_mask) / jnp.sum(atom_mask)
    moment_ctd_2 = jnp.sum((distance_to_ctd - moment_ctd_1)**2 * atom_mask) / jnp.sum(atom_mask)
    moment_ctd_3 = jnp.sum((distance_to_ctd - moment_ctd_1)**3 * atom_mask) / jnp.sum(atom_mask)
    
    cst_idx = jnp.argmin(distance_to_ctd + 1e6 * (1.0 - atom_mask))
    mask_ = atom_mask * (1.0 - jnp.eye(atom_mask.shape[0])[cst_idx])
    distance_to_cst = jnp.linalg.norm(x - x[cst_idx][None, ...] + (1.0 - mask_[..., None]) * 1e-6, axis=-1)
    moment_cst_1 = jnp.sum(distance_to_cst * mask_) / jnp.sum(mask_)
    moment_cst_2 = jnp.sum((distance_to_cst - moment_cst_1)**2 * mask_) / jnp.sum(mask_)
    moment_cst_3 = jnp.sum((distance_to_cst - moment_cst_1)**3 * mask_) / jnp.sum(mask_)
    
    fct_idx = jnp.argmax(distance_to_cst - 1e6 * (1.0 - atom_mask))
    mask_ = atom_mask * (1.0 - jnp.eye(atom_mask.shape[0])[fct_idx])
    distance_to_fct = jnp.linalg.norm(x - x[fct_idx][None, ...] + (1.0 - mask_[..., None]) * 1e-6, axis=-1)
    moment_fct_1 = jnp.sum(distance_to_fct * mask_) / jnp.sum(mask_)
    moment_fct_2 = jnp.sum((distance_to_fct - moment_fct_1)**2 * mask_) / jnp.sum(mask_)
    moment_fct_3 = jnp.sum((distance_to_fct - moment_fct_1)**3 * mask_) / jnp.sum(mask_)
    
    ftf_idx = jnp.argmax(distance_to_fct - 1e6 * (1.0 - atom_mask))
    mask_ = atom_mask * (1.0 - jnp.eye(atom_mask.shape[0])[ftf_idx])
    distance_to_ftf = jnp.linalg.norm(x - x[ftf_idx][None, ...] + (1.0 - mask_[..., None]) * 1e-6, axis=-1)
    moment_ftf_1 = jnp.sum(distance_to_ftf * mask_) / jnp.sum(mask_)
    moment_ftf_2 = jnp.sum((distance_to_ftf - moment_ftf_1)**2 * mask_) / jnp.sum(mask_)
    moment_ftf_3 = jnp.sum((distance_to_ftf - moment_ftf_1)**3 * mask_) / jnp.sum(mask_)
    
    return jnp.array([moment_ctd_1, moment_ctd_2, moment_ctd_3, moment_cst_1, moment_cst_2, moment_cst_3, moment_fct_1, moment_fct_2, moment_fct_3, moment_ftf_1, moment_ftf_2, moment_ftf_3], dtype=jnp.float32)

def moment_calculator(dists, mask, order=3):
    ### dists: (Natom,) mask: (Natom), order: int
    moment_1 = jnp.sum(dists * mask) / jnp.sum(mask)
    moments = [moment_1]
    for o in range(2, order+1):
        moments.append(jnp.sum((dists - moment_1)**o * mask) / jnp.sum(mask))
        # moment_3 = jnp.sum((dists - moment_1)**3 * mask) / jnp.sum(mask)

    return jnp.array(moments, dtype=jnp.float32)

def USR_descriptor_with_extra_centers(x, atom_mask, extra_centers=[]):
    #### x: (Natom), atom_mask: (Natom), extra_centers: (Nc, 3)
    moments_all = []
    
    x, atom_mask = jnp.array(x).astype(jnp.float32), jnp.array(atom_mask).astype(jnp.float32)
    ctd = jnp.sum(x * atom_mask[..., None], axis=0) / jnp.sum(atom_mask)
    distance_to_ctd = jnp.linalg.norm(x - ctd[None, ...] + (1.0 - atom_mask[..., None]) * 1e-6, axis=-1)
    moments_ctd = moment_calculator(distance_to_ctd, atom_mask, order=3)
    moments_all.append(moments_ctd)
    
    cst_idx = jnp.argmin(distance_to_ctd + 1e6 * (1.0 - atom_mask))
    mask_ = atom_mask * (1.0 - jnp.eye(atom_mask.shape[0])[cst_idx])
    distance_to_cst = jnp.linalg.norm(x - x[cst_idx][None, ...] + (1.0 - mask_[..., None]) * 1e-6, axis=-1)
    moments_cst = moment_calculator(distance_to_cst, mask_, order=3)
    moments_all.append(moments_cst)
    
    fct_idx = jnp.argmax(distance_to_cst - 1e6 * (1.0 - atom_mask))
    mask_ = atom_mask * (1.0 - jnp.eye(atom_mask.shape[0])[fct_idx])
    distance_to_fct = jnp.linalg.norm(x - x[fct_idx][None, ...] + (1.0 - mask_[..., None]) * 1e-6, axis=-1)
    moments_fct = moment_calculator(distance_to_fct, mask_, order=3)
    moments_all.append(moments_fct)
    
    ftf_idx = jnp.argmax(distance_to_fct - 1e6 * (1.0 - atom_mask))
    mask_ = atom_mask * (1.0 - jnp.eye(atom_mask.shape[0])[ftf_idx])
    distance_to_ftf = jnp.linalg.norm(x - x[ftf_idx][None, ...] + (1.0 - mask_[..., None]) * 1e-6, axis=-1)
    moments_ftf = moment_calculator(distance_to_ftf, mask_, order=3)
    moments_all.append(moments_ftf)

    if len(extra_centers) > 0:
        extra_centers = jnp.array(extra_centers).astype(jnp.float32)
        ### (1, Natom, 3) - (Nc, 1, 3) -> (Nc, Natom, 3) -> (Nc, Natom)
        distance_to_exc = jnp.linalg.norm(jnp.expand_dims(x, axis=-3) - jnp.expand_dims(extra_centers, axis=-2)\
                                          + (1.0 - atom_mask[None, ..., None]) * 1e-6, axis=-1)
        moments_exc = jax.vmap(partial(moment_calculator, order=3), in_axes=(0, None))(distance_to_exc, atom_mask) ## (Nc, order)
        moments_all.append(moments_exc.reshape(-1))

    return jnp.concatenate(moments_all)

def USR_similarity(structure_1, structure_2, atom_mask_1, atom_mask_2):
    usr_1 = USR_descriptor(structure_1, atom_mask_1)
    usr_2 = USR_descriptor(structure_2, atom_mask_2)

    return 1.0 / (1.0 + jnp.mean(jnp.abs(usr_1 - usr_2)))

def USR_similarity_with_extra_centers(structure_1, structure_2, atom_mask_1, atom_mask_2,
                   extra_centers_1=[], extra_centers_2=[]):
    usr_1 = USR_descriptor_with_extra_centers(structure_1, atom_mask_1, extra_centers_1)
    usr_2 = USR_descriptor_with_extra_centers(structure_2, atom_mask_2, extra_centers_2)

    return 1.0 / (1.0 + jnp.mean(jnp.abs(usr_1 - usr_2)))

def sample_isotropic_GMM(rng_key, centers, stds, centers_mask, n_points):
    rng_key, index_key = jax.random.split(rng_key)
    ids = jax.random.randint(index_key, shape=(n_points,), minval=0, maxval=jnp.sum(centers_mask))
    centers_ = centers[ids]
    stds_ = stds[ids]

    rng_key, normal_key = jax.random.split(rng_key)
    points = centers_ + jax.random.normal(normal_key, shape=(n_points, 3)) * stds_[..., None]
    return points, rng_key

### guidance 2: modified USR shape descriptor
def USR_SN_descriptor(x, stds, atom_mask, rng_key, extra_centers=[]):
    n_points = 1024
    points, _ = sample_isotropic_GMM(rng_key, x, stds, atom_mask, n_points=n_points)

    return USR_descriptor_with_extra_centers(points, jnp.ones(n_points, dtype=jnp.bool_), extra_centers)

def USR_SN_similarity(structure_1, structure_2, stds_1, stds_2, atom_mask_1, atom_mask_2, rng_key, 
                      key_group_ids_1=[], key_group_ids_2=[]):
    rng_key, rng_key_ = jax.random.split(rng_key)
    rng_key_1, rng_key_2 = jax.random.split(rng_key_)

    if len(key_group_ids_1) > 0 and len(key_group_ids_2) > 0:
        key_group_ids_1 = jnp.array(key_group_ids_1, dtype=jnp.int32)
        key_group_ids_2 = jnp.array(key_group_ids_2, dtype=jnp.int32)
        extra_centers_1 = jnp.mean(structure_1[key_group_ids_1], axis=0, keepdims=True)
        extra_centers_2 = jnp.mean(structure_2[key_group_ids_2], axis=0, keepdims=True)
        usr_1 = USR_SN_descriptor(structure_1, stds_1, atom_mask_1, rng_key_1, extra_centers_1)
        usr_2 = USR_SN_descriptor(structure_2, stds_2, atom_mask_2, rng_key_2, extra_centers_2)
    else:
        usr_1 = USR_SN_descriptor(structure_1, stds_1, atom_mask_1, rng_key_1)
        usr_2 = USR_SN_descriptor(structure_2, stds_2, atom_mask_2, rng_key_2)

    return 1.0 / (1.0 + jnp.mean(jnp.abs(usr_1 - usr_2))), rng_key