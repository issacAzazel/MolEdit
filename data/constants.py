import numpy as np

N_MAX_ATOMS = 64
N_PROPERTY_RBFS = 64
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

structure_properties = [
    'rotatable_bonds', 'rings',
    'LogP', 'TPSA', 'DFT_DIPOLE_TOT', 'DFT_HOMO_LUMO_GAP'
]

property_info = {
    'mw': {'mean': 432.58, 'std': 136.44, 'rbf_centers': np.linspace(-3, 3, N_PROPERTY_RBFS), 'rbf_sigma': 0.1}, 
    'rotatable_bonds': {'mean': 8.21, 'std': 5.15, 'rbf_centers': np.linspace(-3, 3, N_PROPERTY_RBFS), 'rbf_sigma': 0.1}, 
    'rings': {'mean': 3.80, 'std': 1.33, 'rbf_centers': np.linspace(-3, 3, N_PROPERTY_RBFS), 'rbf_sigma': 0.1}, 
    'hbond_acceptors': {'mean': 5.60, 'std': 2.52, 'rbf_centers': np.linspace(-3, 3, N_PROPERTY_RBFS), 'rbf_sigma': 0.1}, 
    'hbond_donors': {'mean': 1.88, 'std': 1.89, 'rbf_centers': np.linspace(-3, 3, N_PROPERTY_RBFS), 'rbf_sigma': 0.1}, 
    'LogP': {'mean': 3.74, 'std': 1.84, 'rbf_centers': np.linspace(-3, 3, N_PROPERTY_RBFS), 'rbf_sigma': 0.1}, 
    'TPSA': {'mean': 88.65, 'std': 52.20, 'rbf_centers': np.linspace(-3, 3, N_PROPERTY_RBFS), 'rbf_sigma': 0.1}, 
    'DFT_DIPOLE_TOT': {'mean': 4.65, 'std': 2.30, 'rbf_centers': np.linspace(-3, 3, N_PROPERTY_RBFS), 'rbf_sigma': 0.1}, 
    'DFT_HOMO_LUMO_GAP': {'mean': 0.30, 'std': 0.03, 'rbf_centers': np.linspace(-3, 3, N_PROPERTY_RBFS), 'rbf_sigma': 0.1},
}