import numpy as np
import Xponge
from Xponge.helper import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Lipinski import RotatableBondSmarts
from rdkit.Geometry import Point3D
from rdkit.Chem import rdMolTransforms

from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')

import pickle as pkl 

elements = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P', 
    16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}

def Check_Connectivity(assign, N_atom):
    N = len(assign.bonds.keys())
    if (N < N_atom):
        return False
    visit = np.zeros(N)
    visit[0] = 1
    queue = [0]
    
    count = 0
    while (len(queue) > 0 and count < N + 2):
        queue_tmp = []
        for state in queue:
            for neighbor in assign.bonds[state].keys():
                if (not visit[neighbor]):
                    visit[neighbor] = 1
                    queue_tmp.append(neighbor)
        queue = queue_tmp[:]
        count += 1

    if (np.any(visit == 0)):
        return False
    else:
        return True
    
def assemble_mol_graph(
    atomic_numbers, hydrogen_numbers, structure,
    tolerance = 1.05,
):
    assign = Xponge.Assign()
    for atype, (x, y, z) in zip([elements[i] for i in atomic_numbers], structure):
        assign.add_atom(atype, x=x, y=y, z=z)
    assign.Determine_Connectivity(tolerance=tolerance)
    
    connected = Check_Connectivity(assign, len(atomic_numbers))
    if not connected:
        return False, assign, None
    
    if hydrogen_numbers is not None:
        ### if hydrogen_numbers is None, then coordinates of H is explictly provided in structure
        for ai, (num_H, (x, y, z)) in enumerate(zip(hydrogen_numbers, structure)):
            for i in range(num_H):
                assign.add_atom("H", x=x + (i % 3 == 0), y=y + (i % 3 == 1), z=z + (i % 3 == 2))
                assign.add_bond(ai, assign.atom_numbers - 1, 1)
        
    success = assign.Determine_Bond_Order()
    
    if not success:
        return False, assign, None
    
    try:
        mol =  rdkit.assign_to_rdmol(assign)
        smiles = Chem.MolToSmiles(mol)
        
        return True, assign, smiles
    except:
        return False, assign, None 
    
def check_bonds(bonds, required_bonds, bond_order_map={1: [1,], 2:[2,], 3:[3,], 12: [1,2], -1000: [1,2,3,]},
                allow_perm=True, exchangeable_clusters=[], strict_mode=False, print_info=False):   
    
    num_required_bonds = 0
    num_satisfied_bonds = 0
    for atom_a in required_bonds.keys():
        for atom_b in required_bonds[atom_a].keys():
            num_required_bonds += 1
            allow_atom_b_ids = [atom_a,]
            allow_atom_a_ids = [atom_b,]
            if allow_perm:
                for cluster in exchangeable_clusters:
                    if atom_b in cluster:
                        allow_atom_b_ids.extend(cluster)
                        break
                    if atom_a in cluster:
                        allow_atom_a_ids.extend(cluster)
                        break
                
            # print("{} {}".format(atom_a, atom_b), allow_atom_a_ids, allow_atom_b_ids)
            find_bonds = False
            for a_id in allow_atom_a_ids:
                for b_id in allow_atom_b_ids:
                    if b_id in bonds[a_id].keys():
                        find_bonds = True
                        num_satisfied_bonds += 1
                        break 
                if find_bonds: break

            if print_info and not find_bonds:
                print("Missing {}-{} bond".format(atom_a, atom_b))
            
            if strict_mode and find_bonds:
                num_satisfied_bonds -= 1
                bond_order = required_bonds[atom_a][atom_b]

                correct_bond_order = True
                if atom_b in bonds[atom_a].keys():
                    correct_bond_order = correct_bond_order and (bonds[atom_a][atom_b] in bond_order_map[bond_order])
                elif atom_a in bonds[atom_b].keys():
                    correct_bond_order = correct_bond_order and (bonds[atom_b][atom_a] in bond_order_map[bond_order])
                    
                if not correct_bond_order: 
                    if print_info:
                        print("{}-{} bond order missmatch, require {}, but got {}".format(
                            atom_a, atom_b, bond_order, bonds[atom_a][atom_b]))
                else:
                    num_satisfied_bonds += 1

    return num_required_bonds, num_satisfied_bonds

def get_rotable_bonds(mol):
    rotable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    return rotable_bonds
        
def get_rotable_dihedrals(mol, structure, given_bonds=None, given_rotable_bonds=None):
    conf = mol.GetConformer()
    for i, (x,y,z) in enumerate(structure):
         conf.SetAtomPosition(i,Point3D(float(x), float(y), float(z)))
    bond_ids = np.array([(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()], dtype=np.uint8)
    bond_types = np.array([int(bond.GetBondType()) for bond in mol.GetBonds()], dtype=np.uint8)
    if given_bonds is None:
        bonds = {i: {} for i in range(len(mol.GetAtoms()))}
        for ((atom_i, atom_j), bond_type) in zip(bond_ids, bond_types):
            bonds[atom_i][atom_j] = bonds[atom_j][atom_i] = bond_type
    else:
        bonds = given_bonds

    rotable_bonds = mol.GetSubstructMatches(RotatableBondSmarts) if \
                    given_rotable_bonds is None else given_rotable_bonds
    rotable_bonds_dihedrals = {}
    for (atom_j, atom_k) in rotable_bonds:
        for atom_i in bonds[atom_j].keys():
            if atom_i == atom_k: continue 
            for atom_l in bonds[atom_k].keys():
                if atom_l == atom_j: continue
                rotable_bonds_dihedrals[(atom_i, atom_j, atom_k, atom_l)] = \
                    rdMolTransforms.GetDihedralRad(conf, int(atom_i), int(atom_j), int(atom_k), int(atom_l))
    return rotable_bonds_dihedrals

def uff_optimize(mol, structure):
    conf = mol.GetConformer()
    for i, (x,y,z) in enumerate(structure):
         conf.SetAtomPosition(i,Point3D(float(x), float(y), float(z)))
    mol = Chem.AddHs(mol, addCoords=True)
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int32)

    uff = AllChem.UFFGetMoleculeForceField(mol)
    ene = uff.CalcEnergy()
    force = np.array(uff.CalcGrad(), dtype=np.float32).reshape(-1, 3)
    force = np.delete(force, np.where(atomic_numbers == 1)[0], axis=0)
    
    ## uff optimization
    AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=1000)
    opt_crd = mol.GetConformer().GetPositions().astype(np.float32)
    opt_crd = np.delete(opt_crd, np.where(atomic_numbers == 1)[0], axis=0)
    
    uff = AllChem.UFFGetMoleculeForceField(mol)
    opt_ene = uff.CalcEnergy()
    opt_force = np.array(uff.CalcGrad(), dtype=np.float32).reshape(-1, 3)
    opt_force = np.delete(opt_force, np.where(atomic_numbers == 1)[0], axis=0)

    return ene, force, opt_crd, opt_ene, opt_force