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


##### initialize models (constituents model)
from config.transformer_config import transformer_config
from transformer.model import Transformer, TransformerConfig

with open('params/ZINC_3m/constituents_model/constituents_vocab.pkl', 'rb') as f:
    constituent_vocab_list = pkl.load(f)
NCONSTITUENTS = len(constituent_vocab_list) # 38
NRG_TOKENS = 3 # seq_len = 38 + 3
SEQ_LEN = NCONSTITUENTS + NRG_TOKENS

NRG_VOCABS = 11
transformer_config.deterministic = True
transformer_config.dtype = jnp.float32
transformer = Transformer(
    config=TransformerConfig(
            **{
                **transformer_config,
                "vocab_size": NATOMS + NRG_VOCABS + 1, 
                "output_vocab_size": NATOMS + NRG_VOCABS + 1}, )
)

##### load params
with open("params/ZINC_3m/constituents_model/moledit_params.pkl", "rb") as f:
    params = pkl.load(f)
    params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)

if SHARDING:
    ##### replicate params
    global_sharding = PositionalSharding(jax.devices()).reshape(NDEVICES, 1)
    params = jax.device_put(params, global_sharding.replicate())

def top_p_sampling(logits, rng_key, p=0.9):
    sorted_indices = jnp.argsort(logits)
    sorted_logits = logits[sorted_indices]
    sorted_probs = jax.nn.softmax(sorted_logits)
    cum_probs = jnp.cumsum(sorted_probs)
    invalid_mask = cum_probs < (1-p)
    
    rng_key, sample_key = jax.random.split(rng_key)
    sampled_token = jax.random.categorical(sample_key, sorted_logits+invalid_mask.astype(jnp.float32)*(-1e5))
    
    return sorted_indices[sampled_token], rng_key 
        
##### prepare functions, jit & vmap
jitted_logits_fn = jax.jit(transformer.apply)
top_p_sampling_fn = jax.vmap(jax.vmap(jax.jit(partial(top_p_sampling, p=0.9))))

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

rng_key = jax.random.PRNGKey(42)
from rdkit import Chem 
valence_dict = {
    'C': 4, 'O': 2, 'N': 3, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1, 
    'Si': 4, 'P': 5, 'S': 2
}
valence_dict = {
    Chem.GetPeriodicTable().GetAtomicNumber(k) : v for k, v in valence_dict.items()
}
def tokenize_rg(rg):
    exponents = math.floor(math.log(rg))
    numbers = np.round(rg / math.exp(exponents), 1)
    numbers_str = str(numbers) ### x.yz...
    rg_tokens = [int(exponents), int(numbers_str[0]), int(numbers_str[2])]

    return rg_tokens

sampled_constituents = []

for input_data in tqdm(input_datas):    
    atomic_numbers = input_data['atomic_numbers'].tolist()
    hydrogen_numbers = input_data['hydrogen_numbers'].tolist()
    hybridizations = input_data['hybridizations'].tolist()
    rg = input_data['rg']
    group_constituents_str = ["{}_{}_{}".format(i, j, k) for i, j, k in zip(atomic_numbers, 
                                                                            hydrogen_numbers, 
                                                                            hybridizations)]
    inputs = np.array([group_constituents_str.count(t) for t in constituent_vocab_list] + (np.array(tokenize_rg(rg)) + NATOMS).tolist(), 
                      dtype=np.int32)
    input_dict = {
        "inputs": jnp.array([inputs,] * NBEAMS * NLINKER_PER_MOL, dtype=jnp.int32) + 1, 
        "generation_result": jnp.ones((NBEAMS * NLINKER_PER_MOL, SEQ_LEN), dtype=jnp.int32)
    }
    
    rng_keys = jax.random.split(rng_key, NBEAMS * NLINKER_PER_MOL*SEQ_LEN + 1)
    rng_keys, rng_key = rng_keys[:NBEAMS * NLINKER_PER_MOL*SEQ_LEN].reshape(NBEAMS * NLINKER_PER_MOL, SEQ_LEN, -1), rng_keys[-1]
    
    if SHARDING:
        #### shard inputs 
        ds_sharding = partial(_sharding, shards=global_sharding)
        input_dict = jax.tree_map(ds_sharding, input_dict)
        rng_keys = ds_sharding(rng_keys)
    
    inv_temperature = 1.25
    for step in range(SEQ_LEN):
        logits = jitted_logits_fn(params, 
                                  input_dict['inputs'],
                                  input_dict['generation_result'])
        if step >= NCONSTITUENTS:
            valid_logits_mask = jnp.zeros_like(logits, dtype=jnp.float32).at[..., -NRG_VOCABS:-1].set(1)
        else:
            valid_logits_mask = jnp.zeros_like(logits, dtype=jnp.float32).at[..., 1:-NRG_VOCABS].set(1)
        logits += (-1e5) * (1.0 - valid_logits_mask)
        sampled_token, rng_keys = top_p_sampling_fn(logits * inv_temperature, rng_keys)
        input_dict['generation_result'] = input_dict['generation_result'].at[..., step].set(sampled_token[..., step])
    
    ##### resample radius of gyrations 
    input_dict['generation_result'] = input_dict['generation_result'] + input_dict['inputs'] - 1
    input_dict['inputs'] = jnp.ones_like(input_dict['inputs'], dtype=input_dict['inputs'].dtype)
    input_dict['inputs'] = input_dict['inputs'].at[..., NCONSTITUENTS:].set(NATOMS + NRG_VOCABS) ### unk tokens for rg
    input_dict['generation_result'] = input_dict['generation_result'].at[..., NCONSTITUENTS:].set(NATOMS + NRG_VOCABS)
    inv_temperature = 1.25
    for step in range(NCONSTITUENTS, SEQ_LEN):
        logits = jitted_logits_fn(params, 
                                  input_dict['inputs'],
                                  input_dict['generation_result'])
        if step >= NCONSTITUENTS:
            valid_logits_mask = jnp.zeros_like(logits, dtype=jnp.float32).at[..., -NRG_VOCABS:-1].set(1)
        else:
            valid_logits_mask = jnp.zeros_like(logits, dtype=jnp.float32).at[..., 1:-NRG_VOCABS].set(1)
        logits += (-1e5) * (1.0 - valid_logits_mask)
        sampled_token, rng_keys = top_p_sampling_fn(logits * inv_temperature, rng_keys)
        input_dict['generation_result'] = input_dict['generation_result'].at[..., step].set(sampled_token[..., step])
    
    generation_result = np.array(input_dict['generation_result']) - 1
    generation_result[..., :NCONSTITUENTS] = generation_result[..., :NCONSTITUENTS] - inputs[None, :NCONSTITUENTS]

    num_linker_atoms = np.sum(generation_result[..., :NCONSTITUENTS], axis=-1)
    num_total_atoms = num_linker_atoms + len(atomic_numbers)
    generation_result = generation_result[np.logical_and(num_linker_atoms > 0, num_total_atoms < NATOMS)]
    # print(len(generation_result))
    
    valid_generation_result = []
    ### check valence closure 
    for g in generation_result:
        num_valence = 0
        for i, vocab in enumerate(constituent_vocab_list):
            num_vocab = g[i]
            element, num_H, _ =  vocab.split('_')
            num_valence += num_vocab * (valence_dict[int(element)] - int(num_H))
        # print(num_valence)
        if num_valence % 2 == 0:
            valid_generation_result.append(g)
    valid_generation_result = np.array(valid_generation_result)
    # print(len(valid_generation_result))

    result_strs = ['_'.join([str(x) for x in g[:NCONSTITUENTS]]) for g in valid_generation_result]
    unique_result_strs = {}
    select_ids = []
    for index, result_str in enumerate(result_strs):
        if result_str not in unique_result_strs:
            select_ids.append(index)
            unique_result_strs.update({result_str: None})
    # print(len(select_ids))
    if len(select_ids) < NLINKER_PER_MOL:
        select_ids.extend(np.random.randint(0, len(result_strs), NLINKER_PER_MOL - len(select_ids)))
    else:
        select_ids = np.random.choice(select_ids, size=NLINKER_PER_MOL, replace=False)
    
    constituents = []
    for seq in valid_generation_result[select_ids]:
        #### decode constituents
        atomic_numbers_ = atomic_numbers[:]
        hydrogen_numbers_ = hydrogen_numbers[:]
        hybridizations_ = hybridizations[:]
        for token, num in zip(constituent_vocab_list, seq[:NCONSTITUENTS]):
            atomic_number, hydrogen_number, hybridization = tuple([int(x) for x in token.split('_')])
            atomic_numbers_ = atomic_numbers_ +  [atomic_number,] * num 
            hydrogen_numbers_ = hydrogen_numbers_ + [hydrogen_number,] * num 
            hybridizations_ = hybridizations_ + [hybridization,] * num
            
        #### decode rg
        rg_seq = seq[-NRG_TOKENS:] - NATOMS
        rg_ = np.exp(rg_seq[0]) * float("{}.{}".format(rg_seq[1], "".join([str(x) for x in rg_seq[2:]])))
        constituents.append(
            {"atomic_numbers": np.array(atomic_numbers_, dtype=np.uint8), 
             "hydrogen_numbers": np.array(hydrogen_numbers_, dtype=np.uint8),
             "hybridizations": np.array(hybridizations_, dtype=np.uint8), 
             "radius_of_gyrations": np.array([rg,], dtype=np.float32)} # np.array([rg_,], dtype=np.float32)}
        )
        
    sampled_constituents.append(np.random.choice(constituents, NLINKER_PER_MOL))
    
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