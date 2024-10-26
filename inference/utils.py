import numpy as np
import jax 
import jax.numpy as jnp
from data.constants import feature_all_type
import itertools

def one_hot(depth, indices):
    """one hot compute"""
    res = np.eye(depth)[indices.reshape(-1)]
    return res.reshape(list(indices.shape) + [depth])

def pad_axis(array, n, axis, constant_values=0.0):
    return np.pad(
        array,
        [(0, 0) if i not in axis else (0, n-array.shape[i]) for i in range(array.ndim)],
        mode='constant',
        constant_values=constant_values)

def preprocess_data(input_dict, pad_to_len):
    n_atoms = len(input_dict['atomic_numbers'])
    atom_mask = np.ones(n_atoms, dtype=np.bool_)

    atom_feat = np.concatenate(
        [
            one_hot(len(feature_all_type['atomic_numbers']), 
                    np.array([feature_all_type['atomic_numbers'].index(i) for i in input_dict['atomic_numbers']])),
            one_hot(len(feature_all_type['hybridizations']), 
                    np.array([feature_all_type['hybridizations'].index(i) for i in input_dict['hybridizations']])),
            one_hot(len(feature_all_type['hydrogen_numbers']), 
                    np.array([feature_all_type['hydrogen_numbers'].index(i) for i in input_dict['hydrogen_numbers']]))
        ], axis=-1
    ).astype(np.bool_)

    if 'bonds' not in input_dict.keys():
        bond_feat = np.zeros((n_atoms, n_atoms, 5), dtype=np.bool_)
    else:
        bond_feat = np.ones((n_atoms, n_atoms), dtype=np.int32) * (-1)
        bond_info = input_dict['bonds']
        for atom_a, bond_a in bond_info.items():
            for atom_b, bond_type in bond_a.items():
                bond_feat[atom_a][atom_b] = bond_feat[atom_b][atom_a] \
                    = feature_all_type['bond_types'].index(bond_type)
        bond_feat = np.logical_and(one_hot(5, bond_feat).astype(np.bool_), 
                                   (bond_feat != -1)[...,None])

    rg = np.random.choice(input_dict['radius_of_gyrations'])

    return {
        'atom_mask': pad_axis(atom_mask, pad_to_len, axis=[0], constant_values=0),
        'atom_feat': pad_axis(atom_feat, pad_to_len, axis=[0], constant_values=0),
        'bond_feat': pad_axis(bond_feat, pad_to_len, axis=[0,1], constant_values=0),
        'rg': np.array(rg, dtype=np.float32) 
    }
    
def Kabsch_align(x, y, atom_mask, is_np=True):
    _np = np if is_np else jnp
    _dtype = np.float32 if is_np else jnp.float32
    atom_mask = atom_mask.astype(_dtype)[..., None]
    n_atoms = _np.sum(atom_mask)
    cent_x = _np.sum(x * atom_mask, axis=0, keepdims=True) / n_atoms 
    cent_y = _np.sum(y * atom_mask, axis=0, keepdims=True) / n_atoms
    x = (x - cent_x) * atom_mask
    y = (y - cent_y) * atom_mask
    h = _np.dot(x.T, y)
    u, _, vt = _np.linalg.svd(h)
    d = _np.sign(_np.linalg.det(_np.dot(vt.T, u.T)))
    vt = _np.concatenate([vt[:-1, :], vt[-1:, :] * d], axis=0)
    r = _np.dot(vt.T, u.T)
    x_aligned = _np.dot(x + cent_x, r.T)
    t = - _np.sum(x_aligned * atom_mask, keepdims=True, axis=0) / n_atoms + cent_y
    return x_aligned + t, r.T, t

def permutate(x, perm):
    return x[perm]

def generate_perm_list(N_atom, exchangeable_cluster):    
    N_cluster = len(exchangeable_cluster)
    perm_que = []
    perm_que.append(np.arange(N_atom))

    N_cluster_perm = 1
    cluster_perm_list = []
    for nums in exchangeable_cluster:
        cluster_perm = np.array(list(itertools.permutations(nums)))
        cluster_perm_list.append(cluster_perm)
        N_cluster_perm = N_cluster_perm*len(cluster_perm)

    for cluster in range(N_cluster):
        N_queue = len(perm_que)
        perm_id = exchangeable_cluster[cluster]
        for i in range(N_queue):
            current_perm = perm_que.pop(0)
            for perm in cluster_perm_list[cluster]:
                current_perm_ = current_perm.copy()
                current_perm_[perm_id] = perm.copy()
                perm_que.append(current_perm_)
                if len(perm_que)>500:
                    return perm_que
    
    return perm_que