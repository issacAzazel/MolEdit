import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
import sys 
sys.path.append(".")
import pickle as pkl
import numpy as np
import math
from tqdm import tqdm
print("We are now running Python in: ", sys.path)

import jax 
import jax.numpy as jnp

from functools import partial
from cybertron.common.config_load import load_config

print("########################## PREPARING NETS ################################")

#### load nets
from train.train import MolEditScoreNet
from cybertron.model.molct_plus import MolCT_Plus
from cybertron.readout import GFNReadout

from train.utils import set_dropout_rate_config
from jax.sharding import PositionalSharding

def _sharding(input, shards):

    n_device = shards.shape[0]
    if isinstance(input, (np.ndarray, jax.Array)):
        _shape = [n_device, ] + [1 for _ in range(input.ndim - 1)]
        return jax.device_put(input, shards.reshape(_shape))
    elif input is None:
        return jax.device_put(input, shards)
    else:
        raise TypeError(f"Invalid input: {input}")

from inference.inference import DPM_3_inference, DPM_3_inference_modified_shape_guidance

NDEVICES = 1
NATOMS = 64
SHARDING = True #### you can use multiple devices
FIX_RNGS = True #### To reproduce results
if SHARDING:
    NDEVICES = len(jax.devices())
    print("{} DEVICES detected: {}".format(NDEVICES, jax.devices()))

def split_rngs(rng_key, shape):
    size = np.prod(shape)
    rng_keys = jax.random.split(rng_key, size + 1)
    return rng_keys[:-1].reshape(shape + (-1,)), rng_keys[-1]

rng_key = jax.random.PRNGKey(8888) #### set your random seed here
np.random.seed(7777)


##### initialize models (structure diffusion model)
encoder_config = load_config("config/molct_plus.yaml")
gfn_config = load_config("config/gfn.yaml")
gfn_config.settings.n_interactions = 4

modules = {
    "encoder": {"module": MolCT_Plus, 
                "args": {"config": encoder_config}},
    "gfn": {"module": GFNReadout, 
            "args": {"config": gfn_config}}
}

##### load params
load_ckpt_paths = ['./params/ZINC_3m/structure_model/moledit_params_track1.pkl', 
                   './params/ZINC_3m/structure_model/moledit_params_track2.pkl',
                   './params/ZINC_3m/structure_model/moledit_params_track3.pkl']  
noise_thresholds = [0.35, 1.95]

params = []
for path in load_ckpt_paths:
    with open(path, 'rb') as f: 
        params.append(pkl.load(f))
    
if SHARDING:
    ##### replicate params
    global_sharding = PositionalSharding(jax.devices()).reshape(NDEVICES, 1)
    params = jax.device_put(params, global_sharding.replicate())

for k, v in modules.items():
    modules[k]['args']['config'] = \
        set_dropout_rate_config(modules[k]['args']['config'], 0.0)
    modules[k]["module"] = v["module"](**v["args"])
    modules[k]["callable_fn"] = [] 
    for param in params:
        partial_params = {"params": param["params"]['score_net'].pop(k)}
        modules[k]["callable_fn"].append(partial(modules[k]["module"].apply, partial_params))

moledit_scorenets = [MolEditScoreNet(
        encoder=modules['encoder']['callable_fn'][k],
        gfn=modules['gfn']['callable_fn'][k],
    ) for k in range(len(load_ckpt_paths))]

print('########################## MODEL INITIALIZATION DONE ##########################')

from rdkit import Chem 
def RDMol_to_constituents(mol):
    mol = Chem.RemoveAllHs(mol)
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.uint8)
    hydrogen_numbers = np.array([atom.GetTotalNumHs() + atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()], dtype=np.uint8)
    hybridizations = np.array([atom.GetHybridization() for atom in mol.GetAtoms()], dtype=np.uint8)

    bond_ids = np.array([(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()], dtype=np.uint8)
    bond_types = np.array([int(bond.GetBondType()) for bond in mol.GetBonds()], dtype=np.uint8)
    
    topology = {i: {} for i in range(len(atomic_numbers))}
    for (atom_i, atom_j), bond_type in zip(bond_ids, bond_types):
        topology[atom_i][atom_j] = topology[atom_j][atom_i] = bond_type
    
    constituents_dict = {
        "atomic_numbers": atomic_numbers,
        "hydrogen_numbers": hydrogen_numbers,
        "hybridizations": hybridizations,
        "bonds": topology,
    }

    for c in mol.GetConformers():
        structure = np.array(c.GetPositions())

    return constituents_dict, structure

def SMILES_to_constituents(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.RemoveAllHs(mol)
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.uint8)
    hydrogen_numbers = np.array([atom.GetTotalNumHs() for atom in mol.GetAtoms()], dtype=np.uint8)
    hybridizations = np.array([atom.GetHybridization() for atom in mol.GetAtoms()], dtype=np.uint8)

    bond_ids = np.array([(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()], dtype=np.uint8)
    bond_types = np.array([int(bond.GetBondType()) for bond in mol.GetBonds()], dtype=np.uint8)
    
    topology = {i: {} for i in range(len(atomic_numbers))}
    for (atom_i, atom_j), bond_type in zip(bond_ids, bond_types):
        topology[atom_i][atom_j] = topology[atom_j][atom_i] = bond_type
    
    constituents_dict = {
        "atomic_numbers": atomic_numbers,
        "hydrogen_numbers": hydrogen_numbers,
        "hybridizations": hybridizations,
        "bonds": topology,
    }

    return constituents_dict

#### jit and vmap functions
def score_forward_fn(atom_feat, bond_feat, x, atom_mask, sigma, rg):
    cond_list = [sigma < noise_thresholds[0],] + \
                [jnp.logical_and(sigma >= noise_thresholds[i], sigma < noise_thresholds[i+1]) for i in range(0, len(noise_thresholds) - 1)] + \
                [sigma >= noise_thresholds[-1],]
    value_list = [net.apply(
                    {}, atom_feat, bond_feat, x, atom_mask, sigma, rg)[-1] for net in moledit_scorenets]
    
    return jnp.sum(jnp.array(cond_list, dtype=jnp.float32)[..., None, None] * \
                    jnp.array(value_list, jnp.float32), axis=0)

score_forward_fn_jvj = jax.jit(jax.vmap(jax.jit(score_forward_fn)))

print('########################## PREPARING SHAPE CONSTANTS ##########################')

#### define constants here
idx_to_elements = {
    6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 
    35: 'Br', 53: 'I'
}
vdw_radii = {
    'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.35, 
    'Si': 2.10, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 
    'Br': 1.85, 'I': 1.98
}

NSAMPLE_PER_DEVICE = 128
NSAMPLES = NSAMPLE_PER_DEVICE * NDEVICES

#### Sample atom constituents from datasets
with open('./moledit_dataset/constituents/constituents_ZINC_30m.pkl', 'rb') as f:
    constituents_data = pkl.load(f)

print('# num dataset constituents: {}'.format(len(constituents_data)))

#### entering the main loop
input_dir = "./moledit_dataset/binder_design"
system_dirs = os.listdir(input_dir)
system_dirs.sort()
for i, system_name in enumerate(system_dirs):
    if os.path.exists(f'results/binder_design/results/mol2_{system_name}'):
        continue 
    
    print(f"\n##################### Processing {i} {system_name} ######################\n")
    system_dir = os.path.join(input_dir, system_name)
    sdf_file = os.path.join(system_dir, [x for x in os.listdir(system_dir) if x.endswith('.sdf')][0])
    
    suppl = Chem.SDMolSupplier(sdf_file)
    mol = suppl[0]

    constituents_dict_template, structure_template = RDMol_to_constituents(mol)
    # constituents_dict_template = SMILES_to_constituents('Cc1cc([C@@H](C)Nc2ccccc2C(=O)[O-])c2nc(N3CCCCC3)c(C#N)c(=O)n2c1')
    rg_template = np.sqrt(
        np.mean(np.linalg.norm(structure_template - np.mean(structure_template, axis=0, keepdims=True), axis=-1)**2)
    )
    n_atoms_template = len(structure_template)
    atomic_numbers = constituents_dict_template['atomic_numbers']
    hydrogen_numbers = constituents_dict_template['hydrogen_numbers']
    hybridizations = constituents_dict_template['hybridizations']
    atom_gaussian_stds = np.array([vdw_radii[idx_to_elements[i]] for i in atomic_numbers], dtype=np.float32)
    bonds = constituents_dict_template['bonds']
    print("atomic numbers:  ", ",".join([str(x) for x in atomic_numbers]))
    print("hydrogen_numbers:", ",".join([str(x) for x in hydrogen_numbers]))
    print("hybridizations:  ", ",".join([str(x) for x in hybridizations]))
    print("atom gaussian stds:  ", ",".join([str(x) for x in atom_gaussian_stds]))

    print('===================== Template processing done =====================')

    #### select n_atoms approx templates
    constituents_all = []
    for smi, c in tqdm(constituents_data.items()):
        if np.abs(len(c['atomic_numbers']) - n_atoms_template) < 3:
            c_ = {'radius_of_gyrations': [rg_template, ]}
            c_['atomic_numbers'] = c['atomic_numbers']
            c_['hydrogen_numbers'] = c['hydrogen_numbers']
            c_['hybridizations'] = c['hybridizations']
            if 5 in c_['hybridizations']: continue
            
            constituents_all.append(c_)
                
    print("# selected constituents: {}".format(len(constituents_all)))
    print('===================== Constituents pool processing done =====================')

    print('===================== Inference Started =====================')
        
    random_idx = np.random.randint(0, len(constituents_all), NSAMPLES)
    constituents = [constituents_all[i] for i in random_idx]
    
    print("Example constituents: ")
    print("\tatomic numbers: ", constituents[0]['atomic_numbers'])
    print("\thydrogen numbers: ", constituents[0]['hydrogen_numbers'])
    print("\thybridizaions: ", constituents[0]['hybridizations'])
    print("\tradius of gyrations: ", constituents[0]['radius_of_gyrations'])
    print("\t**REMARK**: hybridization symbols are same with RDkit")

    from inference.utils import preprocess_data

    print("Preprocessing inputs")
    input_dicts = [{**preprocess_data(c, NATOMS), 
                    "gaussian_std": 
        np.pad(np.array([vdw_radii[idx_to_elements[i]] for i in c['atomic_numbers']], dtype=np.float32), (0, NATOMS - len(c['atomic_numbers']))),
    } for c in tqdm(constituents)]
    input_dict = {
        k: np.stack([d[k] for d in input_dicts]) for k in input_dicts[0].keys()
    }

    print("input shape & dtypes: ")
    for k, v in input_dict.items():
        print("\t{} shape: {} dtype: {}".format(k, v.shape, v.dtype))
        
        
    #### make shape dict 
    shape_template_dict = {
        'template_structure': np.pad(structure_template, ((0, NATOMS-n_atoms_template), (0, 0))), 
        'template_key_group_ids': np.array([], dtype=np.int32), 
        'mol_key_group_ids': np.arange(0),
        'template_std': np.pad(atom_gaussian_stds, (0, NATOMS-n_atoms_template)),
        'template_atom_mask': np.pad(np.ones(n_atoms_template, dtype=np.bool_), (0, NATOMS-n_atoms_template)),
        'template_coeff': 0.0,
        'gaussian_scale_factor': 1.0, 
    }

    shape_template_dict['template_structure'] = np.tile(shape_template_dict['template_structure'][None, ...], 
                                                        (NSAMPLES, 1, 1))
    shape_template_dict['template_atom_mask'] = np.tile(shape_template_dict['template_atom_mask'][None, ...], 
                                                        (NSAMPLES, 1))
    shape_template_dict['template_std'] = np.tile(shape_template_dict['template_std'][None, ...], 
                                                (NSAMPLES, 1))
    shape_template_dict['template_key_group_ids'] = np.tile(shape_template_dict['template_key_group_ids'][None, ...], 
                                                            (NSAMPLES, 1))
    shape_template_dict['mol_key_group_ids'] = np.tile(shape_template_dict['mol_key_group_ids'][None, ...], 
                                                    (NSAMPLES, 1))
    
    shape_template_coeff = 32.0 
    shape_gaussian_scale_factor = 1.0
    
    input_dict = jax.tree_map(lambda x:jnp.array(x), input_dict)
    shape_template_dict['template_coeff'] = shape_template_coeff
    shape_template_dict['gaussian_scale_factor'] = shape_gaussian_scale_factor
    shape_template_dict = jax.tree_map(lambda x:jnp.array(x), shape_template_dict)
    #### JAX compiles a jitted function when you call it first time.
    #### so it will be slow when you run this block first time.
    structures, trajectories, rng_key = DPM_3_inference_modified_shape_guidance(input_dict, rng_key, 
                                                        score_forward_fn_jvj, n_steps=20, 
                                                        shard_inputs=True, shape_dict=shape_template_dict)
    
    #### save results 
    suffix = f'{system_name}_usr_sn_coef_{shape_template_coeff}_{shape_gaussian_scale_factor}'

    with open(f'results/binder_design/results_{suffix}.pkl', 'wb') as f: 
        pkl.dump(jax.tree_map(np.array, 
                            {'constituents': constituents,
                            'template': {'template_structure': structure_template, 'template_constituents': constituents_dict_template},
                            'trajectories': trajectories, 'structures': structures}), f)
        
    print('===================== Graph Assembly Started =====================')
    
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*') ### disable rdkit warning
    import Xponge
    from Xponge.helper import rdkit as Xponge_rdkit_helper
    from graph_assembler.graph_assembler import assemble_mol_graph, check_bonds, uff_eval

    structures = np.array(structures)
    success_or_not = []
    smileses = []
    uff_forces = []
    for i, (constituent, structure) in tqdm(enumerate(zip(constituents, structures))):
        atomic_numbers = [int(x) for x in constituent['atomic_numbers']]
        hydrogen_numbers = [int(x) for x in constituent['hydrogen_numbers']]
        structure = structure[:len(atomic_numbers)]
        success, Xponge_mol, smiles = assemble_mol_graph(atomic_numbers, hydrogen_numbers, structure)
        success_or_not.append(success) 
        smileses.append("" if not success else smiles)

        #### export to mol2
        if success:
            ##### delete Hs (Hs are added to help recogonizing topology, their coordinates are fake)
            atoms = Xponge_mol.atoms[::1]
            hydrogen_atom_idx = np.sort([idx for idx, atom in enumerate(atoms) if 'H' in atom])[::-1]
            for atom_idx in hydrogen_atom_idx: 
                Xponge_mol.delete_atom(atom_idx)
            if not os.path.exists(f'results/binder_design/mol2_{suffix}'):
                os.mkdir(f'results/binder_design/mol2_{suffix}')
            Xponge_mol.save_as_mol2(f'results/binder_design/mol2_{suffix}/{i}.mol2', atomtype=None)

            mol = Xponge_rdkit_helper.assign_to_rdmol(Xponge_mol)
            try:
                ene, force, opt_crd, _, _ = uff_eval(mol, structure)
                uff_forces.append(force)
            except:
                uff_forces.append(None)
        else:
            uff_forces.append(None)
    
    print("===================== Resample structures =====================")
    
    constituents_dicts = {}
    ### preprocess smiles
    for i, smi in tqdm(enumerate(smileses)):
        if smi == "": continue
        try:
            constituents_dict = SMILES_to_constituents(smi)
            constituents_dicts[smi] = constituents_dict
        except:
            continue

    print("# valid smileses: {}".format(len(constituents_dicts))) 
    smileses = list(constituents_dicts.keys())
    smileses = smileses + [smileses[-1],] * (NSAMPLES - len(smileses))
    
    print("Example constituents: ")
    print("\tatomic numbers: ", constituents_dicts[smileses[0]]['atomic_numbers'])
    print("\thydrogen numbers: ", constituents_dicts[smileses[0]]['hydrogen_numbers'])
    print("\thybridizaions: ", constituents_dicts[smileses[0]]['hybridizations'])
    print("\tradius of gyrations: ", rg_template)
    print("\tbonds: ", constituents_dicts[smileses[0]]['bonds'])
    print("\t**REMARK**: hybridization symbols are same with RDkit")

    from inference.utils import preprocess_data

    print("Preprocessing inputs...")
    input_dicts = []
    for i, smi in tqdm(enumerate(smileses)):
        d = constituents_dicts[smi]
        try:
            input_dicts.append(preprocess_data({**{k: v for k, v in d.items() if k != 'radius_of_gyrations'}, 
                                                "radius_of_gyrations": [rg_template]}, NATOMS))
        except:
            input_dicts.append(input_dicts[-1])
            smileses[i] = smileses[i-1]
            
            
    input_dict = {
        k: np.stack([d[k] for d in input_dicts]) for k in input_dicts[0].keys()
    }

    print("input shape & dtypes: ")
    for k, v in input_dict.items():
        print("\t{} shape: {} dtype: {}".format(k, v.shape, v.dtype))
        
    input_dict = jax.tree_map(lambda x:jnp.array(x), input_dict)

    inference_fn = partial(DPM_3_inference, score_fn=score_forward_fn_jvj, 
                        n_steps=20, shard_inputs=SHARDING)
    #### JAX compiles a jitted function when you call it first time.
    ### so it will be slow when you run this block first time.
    structures, trajectories, rng_key = inference_fn(input_dict, rng_key)
    
    #### save results 
    with open(f'results/binder_design/results_{suffix}_regen.pkl', 'wb') as f: 
        pkl.dump(jax.tree_map(np.array, 
                            {'smileses': smileses,
                            'constituents': [constituents_dicts[smi] for smi in smileses],
                            'template': {'template_structure': structure_template, 'template_constituents': constituents_dict_template},
                            'trajectories': trajectories, 'structures': structures}), f)
        
    #### load your results 
    with open(f'results/binder_design/results_{suffix}_regen.pkl', 'rb') as f: 
        results = pkl.load(f)
        constituents = results['constituents']
        trajectories = results['trajectories']
        structures = results['structures']
        
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*') ### disable rdkit warning
    import Xponge
    from Xponge.helper import rdkit as Xponge_rdkit_helper
    from graph_assembler.graph_assembler import assemble_mol_graph, check_bonds, uff_eval

    structures = np.array(structures)
    success_or_not = []
    smileses = []
    uff_forces = []
    for i, (constituent, structure) in tqdm(enumerate(zip(constituents, structures))):
        atomic_numbers = constituent['atomic_numbers']
        hydrogen_numbers = constituent['hydrogen_numbers']
        structure = structure[:len(atomic_numbers)]
        success, Xponge_mol, smiles = assemble_mol_graph(atomic_numbers, hydrogen_numbers, structure)
        success_or_not.append(success) 
        smileses.append("" if not success else smiles)

        #### export to mol2
        if success:
            ##### delete Hs (Hs are added to help recogonizing topology, their coordinates are fake)
            atoms = Xponge_mol.atoms[::1]
            hydrogen_atom_idx = np.sort([idx for idx, atom in enumerate(atoms) if 'H' in atom])[::-1]
            for atom_idx in hydrogen_atom_idx: 
                Xponge_mol.delete_atom(atom_idx)
            if not os.path.exists(f'results/binder_design/mol2_{suffix}_regen'):
                os.mkdir(f'results/binder_design/mol2_{suffix}_regen')
            Xponge_mol.save_as_mol2(f'results/binder_design/mol2_{suffix}_regen/{i}.mol2', atomtype=None)

            mol = Xponge_rdkit_helper.assign_to_rdmol(Xponge_mol)
            try:
                ene, force, opt_crd, _, _ = uff_eval(mol, structure)
                uff_forces.append(force)
            except:
                uff_forces.append(None)
        else:
            uff_forces.append(None)
            
    print(f"Generated binders for {system_name} saved in results/binder_design/mol2_{suffix}_regen.")