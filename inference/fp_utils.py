import jax 
import jax.numpy as jnp 
import numpy as np 
from rdkit.Chem.Lipinski import RotatableBondSmarts

def get_rotable_dihedrals(mol):
    bond_ids = np.array([(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()], dtype=np.uint8)
    bond_types = np.array([int(bond.GetBondType()) for bond in mol.GetBonds()], dtype=np.uint8)
    bonds = {i: {} for i in range(len(mol.GetAtoms()))}
    for ((atom_i, atom_j), bond_type) in zip(bond_ids, bond_types):
        bonds[atom_i][atom_j] = bonds[atom_j][atom_i] = bond_type

    rotable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    rotable_bonds_dihedral_ids = []
    for (atom_j, atom_k) in rotable_bonds:
        for atom_i in bonds[atom_j].keys():
            if atom_i == atom_k: continue 
            for atom_l in bonds[atom_k].keys():
                if atom_l == atom_j: continue
                rotable_bonds_dihedral_ids.append([atom_i, atom_j, atom_k, atom_l])
    return np.array(rotable_bonds_dihedral_ids, dtype=np.int32)

def get_dihedral_cos_sin(atom_crd, i, j, k, l):
    crd_i, crd_j, crd_k, crd_l = atom_crd[i], atom_crd[j], atom_crd[k], atom_crd[l]
    a = crd_j-crd_i
    b = crd_k-crd_j
    c = crd_l-crd_k
    n1, n2 = jnp.cross(a, b), jnp.cross(b, c)
    b_ = jnp.cross(n1, n2)
    sign_mask_ = jnp.sum(b*b_, axis=-1)
    sign_mask = sign_mask_ > 0
    
    norm = jnp.maximum(jnp.linalg.norm(n1, axis=-1)*jnp.linalg.norm(n2, axis=-1), 1e-6)

    angles_cos = jnp.clip(jnp.sum(n1 * n2) / norm, -1.0, 1.0)
    angles_sin_candidate_1 = jnp.sqrt(jnp.maximum(1.0 - angles_cos**2, 1e-6))
    angles_sin_candidate_2 = -angles_sin_candidate_1
    angles_sin = jnp.where(sign_mask, angles_sin_candidate_1, angles_sin_candidate_2)
    
    return jnp.array([angles_cos, angles_sin])

def log_gaussian_kernel(x, y, kernel_width, dist_fn):
    dist = dist_fn(x, y)

    return - dist / kernel_width

def fp_dihedral_gradient(atom_crd, dihedral_index):
    # atom_crd: [Nconf, Natoms, 3], dihedral index: [Ndihedral, 4]

    ### get kernel width
    get_dihedral_cos_sin_vv = jax.vmap(jax.vmap(get_dihedral_cos_sin, in_axes=(None, 0, 0, 0, 0)), in_axes=(0, None, None, None, None))
    dihedral_cos_sins = get_dihedral_cos_sin_vv(atom_crd, dihedral_index[:, 0], dihedral_index[:, 1], dihedral_index[:, 2], dihedral_index[:, 3]) ### (Nconf, Ndihedral, 2)
    dist = 2.0 - 2.0 * jnp.sum(dihedral_cos_sins[None,...] * dihedral_cos_sins[:,None,...], axis=-1) ### (Nconf, Nconf, Ndihedral)
    dist_ = dist.reshape(-1, dist.shape[-1])
    kernel_width = jax.lax.stop_gradient(jnp.median(dist_, axis=0)) + 1e-3 ## (Ndihedral,)

    get_dihedral_cos_sin_v = jax.vmap(get_dihedral_cos_sin, in_axes=(None, 0, 0, 0, 0))
    def kernel_fn(xi, xj):
        # xi: (Natom, 3,), xj: (Natom, 3)
        dihedral_cos_sin_i = get_dihedral_cos_sin_v(xi, dihedral_index[:, 0], dihedral_index[:, 1], dihedral_index[:, 2], dihedral_index[:, 3]) ### (Ndihedral, 2)
        dihedral_cos_sin_j = get_dihedral_cos_sin_v(xj, dihedral_index[:, 0], dihedral_index[:, 1], dihedral_index[:, 2], dihedral_index[:, 3]) ### (Ndihedral, 2)

        log_kij_d = log_gaussian_kernel(dihedral_cos_sin_i, dihedral_cos_sin_j, kernel_width, dist_fn = lambda x,y: 2.0 - 2.0 * jnp.sum(x*y, axis=-1)) ### (Ndihedral)
        log_kij = jnp.sum(log_kij_d)
        return jnp.exp(log_kij)

    value_grad_kernel_fn = jax.value_and_grad(kernel_fn)
    value_grad_kernel_fn_vv = jax.vmap(jax.vmap(value_grad_kernel_fn, in_axes=(None, 0)), in_axes=(0, None))

    kij, nabla_kij = value_grad_kernel_fn_vv(atom_crd, atom_crd) ### (Nconf, Nconf), (Nconf, Nconf, Natom, 3)
    return jnp.mean(nabla_kij, axis=1) / (jnp.sum(kij, axis=-1)[..., None, None] + 1e-6)