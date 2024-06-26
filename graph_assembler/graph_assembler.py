import numpy as np
import Xponge
from Xponge.helper import rdkit
from rdkit import Chem

import pickle as pkl 

elements = {
    6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 
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
