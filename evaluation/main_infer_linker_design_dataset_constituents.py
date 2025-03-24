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

from inference.inference import DPM_3_inference, Langevin_inference, DPM_pp_2S_inference

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

print("########################## LOAD INPUT DATA ################################")

with open(f'./moledit_dataset/linker_design/ZINC_test.pkl', 'rb') as f:
    raw_datas = pkl.load(f)[:2]
# with open(f'./moledit_dataset/linker_design/CASF_test.pkl', 'rb') as f:
#     raw_datas = pkl.load(f)

NLINKER_PER_MOL_TEST = 250
NLINKER_PER_MOL = 256
NBEAMS = 8

input_datas = []

def radius_of_gyration(structure):
    structure = structure - np.mean(structure, axis=0, keepdims=True)

    return np.sqrt(np.sum(structure**2, axis=-1).mean(-1))

for raw_data in tqdm(raw_datas):
    input_atom_ids = np.where(raw_data['raw_data']['fragment_mask'])[0]
    input_data = {}

    input_data['atomic_numbers'] = raw_data['constituent']['atomic_numbers'][input_atom_ids]
    input_data['hybridizations'] = raw_data['constituent']['hybridizations'][input_atom_ids]
    input_data['hydrogen_numbers'] = raw_data['constituent']['hydrogen_numbers'][input_atom_ids]
    raw_bonds = raw_data['constituent']['bonds']

    atom_maps = {atom_id: i for i, atom_id in enumerate(input_atom_ids)}
    input_data['bonds'] = {
        atom_maps[atom_i]: {atom_maps[atom_j]: bond_order for atom_j, bond_order in raw_bonds[atom_i].items() if atom_j in atom_maps} for atom_i in raw_bonds.keys()
        if atom_i in atom_maps
    }
    input_data['structures'] = raw_data['raw_data']['positions'][input_atom_ids]
    input_data['rg'] = radius_of_gyration(raw_data['raw_data']['positions'])
    input_data['template_linker_size'] = np.sum(raw_data['raw_data']['linker_mask'])
    
    input_datas.append(input_data)

result_dir = 'results/linker_design/ZINC' # 'results/linker_design/CASF'
os.makedirs(result_dir, exist_ok=True)
with open(f'{result_dir}/input_datas.pkl', 'wb') as f:
    pkl.dump(input_datas, f)
    
print("########################## CONSTITUENTS SAMPLING ################################")

#### load & build fragment library
from rdkit import Chem 

with open('./moledit_dataset/linker_design/frag_constituents.pkl', 'rb') as f:
    frag_lib = pkl.load(f)

fragments = []
probs = []
for k, v in frag_lib.items():
    fragments.append(k)
    probs.append(v)

mean_frag_size = 0
for k, v in tqdm(frag_lib.items()):
    mol = Chem.RemoveAllHs(Chem.MolFromSmiles(k.replace('[X*]', '[H]')))
    mean_frag_size += len(mol.GetAtoms()) * v
    
def parse_frags(frags, min_atoms=4):
    if len(frags) == 0: return None
    n_breakpoints = np.sum([f.count('[X*]') for f in frags])
    if n_breakpoints % 2 == 0 and n_breakpoints >= 2 + (len(frags) - 1) * 2:
        ### get constituents
        atomic_numbers, hybridizations, hydrogen_numbers = [], [], []
        for frag in frags:
            frag_mol = Chem.MolFromSmiles(frag.replace('[X*]', '[1*]'))
            for atom in frag_mol.GetAtoms():
                if atom.GetAtomicNum() != 0:
                    atomic_numbers.append(atom.GetAtomicNum())
                    hybridizations.append(atom.GetHybridization())
                    hydrogen_numbers.append(atom.GetTotalNumHs())
        if len(atomic_numbers) >= min_atoms:
            return {'atomic_numbers': atomic_numbers, 'hybridizations': hybridizations, 'hydrogen_numbers': hydrogen_numbers}
        else:
            return None
    else:
        return None

from multiprocessing import Pool

sampled_constituents = []
for i, input_data in tqdm(enumerate(input_datas)):
    constituents = []
    n_linker_template = input_data['template_linker_size']
    n_mean_frags = n_linker_template / mean_frag_size
    
    def sample_a_constituent(_):
        np.random.seed(_ * 997 + i * 101)
        while (True):
            n_frags = np.random.poisson(n_mean_frags, )
            sampled_frags = np.random.choice(fragments, p=probs, size=(n_frags, ))
            constituent = parse_frags(sampled_frags)
            if constituent is not None:
                return constituent

    # for _ in tqdm(range(NBEAMS * NLINKER_PER_MOL)):
        # while (True):
        #     n_frags = np.random.poisson(n_mean_frags, )
        #     sampled_frags = np.random.choice(fragments, p=probs, size=(n_frags, ))
        #     constituent = parse_frags(sampled_frags)
        #     if constituent is not None:
        #         constituents.append(constituent)
        #         break
    
    with Pool(64) as p:
        constituents = list((tqdm(p.imap(sample_a_constituent, range(NBEAMS * NLINKER_PER_MOL)),
                                  total = NBEAMS * NLINKER_PER_MOL)))
            
    n_atoms = [len(c['atomic_numbers']) for c in constituents]
    ### resample constituents according to normal distribution 
    constituents_size = {k: {} for k in n_atoms}

    def zip_constituent(c):
        return "/".join([str(x) for x in np.sort(['{}_{}_{}'.format(int(i), int(j), int(k)) for i, j, k in zip(
            c['atomic_numbers'], c['hybridizations'], c['hydrogen_numbers']
        )])])

    def unzip_constituent(c_str):
        strs = c_str.split('/')
        c = {
            'atomic_numbers': [], 'hydrogen_numbers': [], 'hybridizations': [], 
        }
        for s in strs:
            i, j, k = s.split('_')
            c['atomic_numbers'].append(int(i))
            c['hybridizations'].append(int(j))
            c['hydrogen_numbers'].append(int(k))
        return c

    def greedy_sample(choices, k):
        if isinstance(choices, np.ndarray): choices = choices.tolist()
        if k < len(choices):
            return np.random.choice(choices, replace=False, size=k).tolist()
        else:
            return choices + np.random.choice(choices, size=k-len(choices)).tolist()

    for c in constituents:
        constituents_size[len(c['atomic_numbers'])].update({zip_constituent(c):None})

    n_atoms = np.unique(n_atoms)
    rescaled_probs = np.exp(-1.0 / 1.0 * 0.5 * (n_atoms - n_linker_template) ** 2) # 3 sigma
    n_atoms_resampled = np.random.choice(n_atoms, p=rescaled_probs/np.sum(rescaled_probs), size=NLINKER_PER_MOL)

    constituents_resampled = []
    for n_atom in n_atoms:
        sampled_size = np.sum(n_atoms_resampled == n_atom)
        constituents_resampled.extend([unzip_constituent(c) for c in greedy_sample(list(constituents_size[n_atom].keys()), sampled_size)])
        
    constituents_ = [] 
    for c in constituents_resampled:
        # atomic_numbers = input_data['atomic_numbers'].tolist() + c['atomic_numbers']
        c_ = {k: input_data[k].tolist() + v for k, v in c.items()}
        c_['radius_of_gyrations'] = [input_data['rg'], ]
        constituents_.append(c_)

    sampled_constituents.append(constituents_) 
    
with open(f'{result_dir}/sampled_constituents.pkl', 'wb') as f:
    pkl.dump(sampled_constituents, f)

print("########################## STRUCTURE SAMPLING ################################")

rng_key = jax.random.PRNGKey(2025)

NATOMS = 64 
INFERENCE_METHOD = "DPM_3"

#### jit and vmap functions
def score_forward_fn(atom_feat, bond_feat, x, atom_mask, sigma, rg, gamma=1.0):
    cond_list = [sigma < noise_thresholds[0],] + \
                [jnp.logical_and(sigma >= noise_thresholds[i], sigma < noise_thresholds[i+1]) for i in range(0, len(noise_thresholds) - 1)] + \
                [sigma >= noise_thresholds[-1],]
    value_list = [net.apply(
                    {}, atom_feat, bond_feat, x, atom_mask, sigma, rg)[-1] for net in moledit_scorenets]
    value_unc_list = [net.apply(
                    {}, atom_feat, jnp.zeros_like(bond_feat), x, atom_mask, sigma, rg)[-1] for net in moledit_scorenets]
    value = gamma * jnp.array(value_list, jnp.float32) +\
                (1.0 - gamma) * jnp.array(value_unc_list, jnp.float32)
    
    return jnp.sum(jnp.array(cond_list, dtype=jnp.float32)[..., None, None] * value, axis=0)

score_forward_fn_jvj = jax.jit(jax.vmap(jax.jit(score_forward_fn)))
if INFERENCE_METHOD == "DPM_3":
    inference_fn = partial(DPM_3_inference, score_fn=score_forward_fn_jvj, 
                           n_steps=20, n_eq_steps=50, shard_inputs=SHARDING)
elif INFERENCE_METHOD == "DPM_pp_2S":
    inference_fn = partial(DPM_pp_2S_inference, score_fn=score_forward_fn_jvj, 
                           n_steps=20, shard_inputs=SHARDING)
elif INFERENCE_METHOD == "Langevin":
    inference_fn = partial(Langevin_inference, score_fn=score_forward_fn_jvj, 
                           n_steps=1000, shard_inputs=SHARDING)
    
from inference.utils import preprocess_data

for i, (input_data, sampled_constituent) in enumerate(zip(input_datas, sampled_constituents)):
    try:
        print("Preprocessing {}".format(i))
        input_dicts = [preprocess_data({**c, "bonds": input_data['bonds']}, NATOMS) for c in tqdm(sampled_constituent)]
        input_dict = {
            k: np.stack([d[k] for d in input_dicts]) for k in input_dicts[0].keys()
        }

        repaint_info = {
            "structure": np.pad(np.array(input_data['structures']), ((0,NATOMS - len(input_data['structures'])), (0,0))).astype(np.float32),
            "mask": np.array([True if i < len(input_data['structures']) else False for i in range(NATOMS)]).astype(np.bool_), 
        }

        repaint_dict = jax.tree_map(lambda x:np.repeat(x[None, ...], NLINKER_PER_MOL, axis=0), repaint_info)
            
        input_dict = jax.tree_map(lambda x:jnp.array(x), input_dict)
        repaint_dict = jax.tree_map(lambda x:jnp.array(x), repaint_dict)
        structures, trajectories, rng_key = inference_fn(input_dict, rng_key, repaint_dict=repaint_dict)
        
        #### save results 
        os.makedirs(f'{result_dir}/structures', exist_ok=True)
        with open(f'{result_dir}/structures/result_{i}.pkl', 'wb') as f: 
            pkl.dump(jax.tree_map(np.array, 
                                {'constituents': sampled_constituent, 'structures': structures}), f)
    except Exception as e:
        print("Error: {}".format(str(e)))