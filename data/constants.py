import numpy as np

N_MAX_ATOMS = 64
UINT8_MAX = np.iinfo(np.uint8).max

######## constants
feature_all_type = {'atomic_numbers': [6, 7, 8, 9, 14, 15, 16, 17, 35, 53],
            'hybridizations': [2, 3, 4],
            'hydrogen_numbers': [0, 1, 2, 3],
            'bond_types': [1, 2, 3, 12, UINT8_MAX],
            }

moledit_data_column_dtypes = {
    'atom_mask': np.bool_,
    'atom_feat': np.bool_,
    'bond_feat': np.bool_,
    'input_structures': np.float32,
    'noise_scale': np.float32,
    'labels': np.float32,
    'rg': np.float32
}

mask_types = ['no_mask', 'atom_mask_random_walk', 'bond_mask_brics', 'bond_mask_random_walk', 'all_mask']