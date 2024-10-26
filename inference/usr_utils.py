import jax
import jax.numpy as jnp

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

def USR_similarity(structure_1, structure_2, atom_mask_1, atom_mask_2):
    usr_1 = USR_descriptor(structure_1, atom_mask_1)
    usr_2 = USR_descriptor(structure_2, atom_mask_2)

    return 1.0 / (1.0 + jnp.mean(jnp.abs(usr_1 - usr_2)))