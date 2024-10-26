import numpy as np
import math
import itertools
from rdkit import Chem

def is_valid_perm(perm, topology):
    for i, perm_i in enumerate(perm):
        for j, bond_ij in topology[i].items():
            perm_j = perm[j]
            if perm_j in topology[perm_i].keys() and bond_ij == topology[perm_i][perm_j]:
                continue 
            else:
                return False
    return True

def generate_perm_list(n_atoms, exchangeable_cluster, max_perms=65536):    
    n_clusters = len(exchangeable_cluster)
    perm_que = []
    perm_que.append(np.arange(n_atoms))

    n_cluster_perm = 1
    cluster_perm_list = []
    for nums in exchangeable_cluster:
        if n_cluster_perm > max_perms:
            return None
        cluster_perm = np.array(list(itertools.permutations(nums)))
        cluster_perm_list.append(cluster_perm)
        n_cluster_perm = n_cluster_perm*len(cluster_perm)

    for cluster in range(n_clusters):
        n_queue = len(perm_que)
        perm_id = exchangeable_cluster[cluster]
        for i in range(n_queue):
            current_perm = perm_que.pop(0)
            for perm in cluster_perm_list[cluster]:
                current_perm_ = current_perm.copy()
                current_perm_[perm_id] = perm.copy()
                perm_que.append(current_perm_)
                if len(perm_que) > max_perms:
                    return None
    return perm_que

def get_valid_perms(atom_types, topology=None, max_perms=65536):
    n_atoms = len(atom_types)
    num_extend_rounds = 4
    if topology is not None:
        for round in range(num_extend_rounds):
            #### get degree profile 
            degree_profile = ["/".join(np.sort(["{}_{}".format(atom_types[k], v) for k, v in topology[i].items()])) for i in range(n_atoms)]
            # degree_profile = ["/".join(np.sort(["{}".format(v) for v in topology[i].values()])) for i in range(n_atoms)]
            extended_atom_types = np.array(["{}_dp_{}".format(a, d) for a, d in zip(atom_types, degree_profile)])
            atom_types = extended_atom_types
    
    #### get exchangeable clusters
    unique_atom_type = np.unique(atom_types)
    exchangeable_clusters = []
    for atom_type in unique_atom_type:
        exchangeable_clusters.append(np.where(extended_atom_types == atom_type)[0])
    num_perms = 1
    for c in exchangeable_clusters:
        num_perms *= math.factorial(len(c))
        if num_perms > max_perms: return None
    
    all_perms = generate_perm_list(n_atoms, exchangeable_clusters)
    if all_perms is None: return None #### too many permutations!
    valid_perms = [perm for perm in all_perms if is_valid_perm(perm, topology)]
    
    return valid_perms

def get_valid_perms_from_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.RemoveAllHs(mol)
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.uint8)
    hydrogen_numbers = np.array([atom.GetTotalNumHs() for atom in mol.GetAtoms()], dtype=np.uint8)
    hybridizations = np.array([atom.GetHybridization() for atom in mol.GetAtoms()], dtype=np.uint8)

    bond_ids = np.array([(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()], dtype=np.uint8)
    bond_types = np.array([int(bond.GetBondType()) for bond in mol.GetBonds()], dtype=np.uint8)
    
    atom_types = ["{}_{}_{}".format(i, j, k) for i, j, k in zip(atomic_numbers, hydrogen_numbers, hybridizations)]
    topology = {i: {} for i in range(len(atomic_numbers))}
    for (atom_i, atom_j), bond_type in zip(bond_ids, bond_types):
        topology[atom_i][atom_j] = topology[atom_j][atom_i] = bond_type
    valid_perms = get_valid_perms(atom_types, topology)
    
    return valid_perms