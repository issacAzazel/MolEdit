import numpy as np 
import pickle as pkl
import os
import concurrent
from data.utils import pad_axis

from data.constants import N_MAX_ATOMS
from data.constants import moledit_data_column_dtypes, mask_types, feature_all_type
from data.utils import pad_axis, one_hot

def read_file(file_path):
    with open(file_path, 'rb') as f:
        return pkl.load(f)
    
def read_files_in_parallel(file_paths, num_parallel_worker=32):
    # time0 = datetime.datetime.now()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_worker) as executor:
        # Map the read_file function to each file path
        results = list(executor.map(read_file, file_paths))
    return results

def process_structure_batch(data_structure_batch, n_samples_per_structure):
    random_select_noise_idx = [np.random.randint(0, len(d_k["noise_scale"]))
                               for d_k in data_structure_batch]
    random_select_structure_idx = [x // n_samples_per_structure for x in random_select_noise_idx]
    
    batch_structure = {}
    batch_structure['rg'] = [d['rg'][i] for i, d in 
                                      zip(random_select_structure_idx, data_structure_batch)]
    batch_structure['input_structures'] = [d['perm_transrot_crd'][i] for i, d in 
                                                   zip(random_select_noise_idx, data_structure_batch)]
    batch_structure['labels'] = [d['perm_transrot_label'][i] for i, d in 
                                          zip(random_select_noise_idx, data_structure_batch)]
    batch_structure['noise_scale'] = [d['noise_scale'][i] for i, d in 
                                          zip(random_select_noise_idx, data_structure_batch)]
    
    #### padding and stack 
    batch_structure['rg'] = np.stack(batch_structure['rg'])
    batch_structure['noise_scale'] = np.stack(batch_structure['noise_scale'])
    batch_structure['input_structures'] = np.stack([pad_axis(x, N_MAX_ATOMS, axes=(0,))
                                                    for x in batch_structure['input_structures']])
    batch_structure['labels'] = np.stack([pad_axis(x, N_MAX_ATOMS, axes=(0,))
                                          for x in batch_structure['labels']])
    
    return batch_structure

def unflatten_bond_matrix(bond_ids, current_bond_feats, feat_name):
    bond_feature = np.ones((N_MAX_ATOMS, N_MAX_ATOMS), dtype=np.int32) * (-1)
    bond_ids = bond_ids
    
    for (atom_i, atom_j), bond_feat in zip(bond_ids, current_bond_feats):
        bond_feature[atom_i][atom_j] = bond_feature[atom_j][atom_i] = \
            feature_all_type[feat_name].index(int(bond_feat))
    
    return bond_feature

def process_feature_batch(data_feature_batch):
    batch_feature = {}
    batch_feature['atom_mask'] = [np.ones_like(d['atomic_numbers'], dtype=np.bool_) 
                                  for d in data_feature_batch]
    batch_feature['atom_mask'] = np.stack([pad_axis(x, N_MAX_ATOMS, axes=(0,))
                                           for x in batch_feature['atom_mask']])
    
    atom_features = []
    for feat_name in ['atomic_numbers', 'hybridizations', 'hydrogen_numbers']:
        current_feature = [
            np.array([feature_all_type[feat_name].index(int(x))
                      for x in d[feat_name]], dtype=np.int32) for d in data_feature_batch 
        ] 
        current_feature = [one_hot(len(feature_all_type[feat_name]), x) for x in current_feature]
        current_feature = [pad_axis(x, N_MAX_ATOMS, axes=(0,)) for x in current_feature]
        atom_features.append(np.stack(current_feature))
    batch_feature['atom_feat'] = np.concatenate(atom_features, axis=-1)
    
    bond_features = []
    for feat_name in ['bond_types']:
        current_feature = [
            unflatten_bond_matrix(d['bond_ids'], d[feat_name], feat_name) for d in data_feature_batch
        ]
        current_feature = [
            np.logical_and(one_hot(len(feature_all_type[feat_name]), x),
                           x[..., None] != -1) for x in current_feature]
        current_feature = [pad_axis(x, N_MAX_ATOMS, axes=(0,1)) for x in current_feature]
        bond_features.append(np.stack(current_feature))
    batch_feature['bond_feat'] = np.concatenate(bond_features, axis=-1)
    
    return batch_feature

def load_train_data_pickle(name_list, 
                           start_idx, 
                           end_idx,
                           num_parallel_worker=32,
                           n_samples_per_structure=4,
                           feature_processed=False,
                           atom_feat_dim=17, 
                           bond_feat_dim=5,
                           allowed_mask_types=mask_types):
    
    pkl_paths = name_list[start_idx: end_idx]
    data_batch = read_files_in_parallel(pkl_paths, num_parallel_worker=num_parallel_worker)
    random_select_mask_types = np.random.choice(allowed_mask_types, len(data_batch))
    
    data_feature_batch = [d['feature'][t] for d, t in zip(data_batch, random_select_mask_types)]
    data_structure_batch = [d['structure'][t] for d, t in zip(data_batch, random_select_mask_types)]
    # for k, v in data_structure_batch[0].items():
    #     print(k, v.shape)
    
    batch_structure_data = process_structure_batch(data_structure_batch, n_samples_per_structure)
    if not feature_processed:
        batch_feature_data = process_feature_batch(data_feature_batch)
    else:
        atom_masks = [np.ones(d['n_atoms'], dtype=np.bool_) for d in data_feature_batch]
        atom_feats = [np.unpackbits(d['atom_feat'])[:d['n_atoms']*atom_feat_dim].reshape(d['n_atoms'], -1) for d in data_feature_batch]
        bond_feats = [np.unpackbits(d['bond_feat'])[:d['n_atoms']*d['n_atoms']*bond_feat_dim].reshape(d['n_atoms'], d['n_atoms'], -1) for d in data_feature_batch]
        batch_feature_data = {
            'atom_mask': np.stack([pad_axis(f, N_MAX_ATOMS, axes=(0,)) for f in atom_masks]),
            'atom_feat': np.stack([pad_axis(f, N_MAX_ATOMS, axes=(0,)) for f in atom_feats]),
            'bond_feat': np.stack([pad_axis(f, N_MAX_ATOMS, axes=(0,1)) for f in bond_feats])
        }
    
    batch_data = {**batch_structure_data, **batch_feature_data}
    
    for k, v in batch_data.items():
        batch_data[k] = v.astype(moledit_data_column_dtypes[k])
    
    return batch_data      