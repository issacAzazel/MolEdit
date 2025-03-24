import jax
import jax.numpy as jnp
import pickle as pkl
import numpy as np
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Inputs for main.py')
    parser.add_argument('--train_config', default="./config/train.yaml", help='train config')
    parser.add_argument('--gfn_config', default="./config/gfn.yaml", help='GFN config')
    # parser.add_argument('--molct_plus_config', default="./config/molct_plus.yaml", help='molct plus config')
    parser.add_argument('--encoder_config', default="./config/molct_plus.yaml", help='molct plus config')
    parser.add_argument('--name_list_pkl_path', help="name list pickle path")
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='gradient clip value')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay value')
    parser.add_argument('--logger_path', type=str, default="./logs/log_moledit.txt", help='Location of logger file.')
    parser.add_argument('--load_ckpt_path', type=str, help='Location of checkpoint file.')
    parser.add_argument('--load_opt_state_path', type=str, help='Location of opt_state file')
    parser.add_argument('--save_ckpt_dir', type=str, default="./moledit_params", help='Location of checkpoint file.')
    parser.add_argument('--logger_step', type=int, default=10, help='logger step')
    parser.add_argument('--save_ckpt_step', type=int, default=2000, help='save ckpt step')
    parser.add_argument('--random_seed', type=int, default=8888, help='random seed')
    parser.add_argument('--start_step', type=int, default=0, help='start step')
    
    parser.add_argument('--n_samples_per_structure', type=int, required=True)
    
    parser.add_argument('--coordinator_address', type=str, help="coordinator address")
    parser.add_argument('--num_processes', type=int, default=2, help="number of processes")
    parser.add_argument('--rank', type=int, default=0, help="rank")
    
    parser.add_argument('--stage', type=int, help='stage')
    parser.add_argument('--allowed_mask_types', nargs='+', type=str, default=['all_mask', 'no_mask', 'bond_mask_random_walk', 'atom_mask_random_walk', 'bond_mask_brics'])
    
    arguments = parser.parse_args()
    return arguments

args = arg_parse()

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
jax.distributed.initialize(
    coordinator_address=args.coordinator_address,
    num_processes=args.num_processes,
    process_id=args.rank,
    local_device_ids=[0,1,2,3,4,5,6,7]
)
RANK = jax.process_index()
print("Multiprocess initializing, Hello from ", RANK)

from functools import partial
from jax import value_and_grad, jit, vmap
from optax import adam
from flax import linen as nn
from flax import traverse_util
from cybertron.common.config_load import load_config
from cybertron.model.molct_plus import MolCT_Plus
from cybertron.readout import GFNReadout
from train.train import MolEditScoreNet, MolEditWithVELossCell, moledit_ve_forward_per_device
from train.utils import logger, set_dropout_rate_config, parameter_weight_decay, any_nan_in_tree, orgnize_name_list
from train.sharding import _sharding
from data.dataset import load_train_data_pickle
import datetime
import optax
from flax.jax_utils import replicate
from jax.sharding import PositionalSharding

from config.global_setup import EnvironConfig
global_setup = EnvironConfig() ### Set Hyper-parameters here
SHARDING = global_setup.sharding
USE_DROPOUT = global_setup.use_dropout
GLOBAL_DROPOUT_RATE = 0.0 if not USE_DROPOUT else global_setup.global_dropout_rate

moledit_input_key_names = ["atom_feat", "bond_feat", "input_structures", "atom_mask", "noise_scale", "labels", "rg"]

loss_name_dict = {
    "mse_last_iter": 0,
    "weight_decay": 0,
}

def train():
    ##### constants 
    NATOM = 64
    NSAMPLE_PER_DEVICE = 16
    NGLOBAL_DEVICES = len(jax.devices())
    NLOCAL_DEVICES = len(jax.local_devices())
    GLOBAL_BATCH_SIZE = NSAMPLE_PER_DEVICE * NGLOBAL_DEVICES
    LOCAL_BATCH_SIZE = NSAMPLE_PER_DEVICE * NLOCAL_DEVICES
    NEPOCH = 50000
    
    np.random.seed(args.random_seed)
    rng_key = jax.random.PRNGKey(args.random_seed)
    
    with open(args.name_list_pkl_path, 'rb') as f:
        name_list_bin = pkl.load(f)
    
    # DATA_BINS = [(0,15), (15, 23), (23, 30), (30, 60)]
    DATA_BINS = list(name_list_bin.keys())
    NUM_BINS = len(DATA_BINS)   
    NDATAS = np.sum([name_list_bin[b]['size'] for b in DATA_BINS])
    STEPS_PER_EPOCH = NDATAS // GLOBAL_BATCH_SIZE
    TOTAL_STEP = STEPS_PER_EPOCH * NEPOCH
    ###### prevent index error
    # NDATAS = STEPS_PER_EPOCH * BATCH_SIZE
    
    print("DATA: ")
    print("\tNRES: {}".format(NATOM))
    print("\tNDATAS: {}".format(NDATAS))
    print("\tBATCH_SIZE: {}".format(GLOBAL_BATCH_SIZE))
    print("\tSTEPS_PER_EPOCH: {}".format(STEPS_PER_EPOCH))
    
    ##### initialize models & load_configs
    encoder_config = load_config(args.encoder_config)
    gfn_config = load_config(args.gfn_config)
    train_config = load_config(args.train_config)
    
    # ##### stage 1:
    # gfn_config.settings.n_interactions = 1
    # train_config.iter_weights = [1.0, ]
    # ##### stage 2:
    # gfn_config.settings.n_interactions = 2
    # train_config.iter_weights = [0.5, 1.0]
    # #### stage 3:
    # gfn_config.settings.n_interactions = 3
    # train_config.iter_weights = [0.25, 0.5, 1.0]
    # ##### stage 4:
    # gfn_config.settings.n_interactions = 4
    # train_config.iter_weights = [0.125, 0.25, 0.5, 1.0]
    
    gfn_config.settings.n_interactions = args.stage 
    if args.stage == 1:
        train_config.iter_weights = [1.0, ]
    elif args.stage == 2:
        train_config.iter_weights = [0.5, 1.0]
    elif args.stage == 3:
        train_config.iter_weights = [0.25, 0.5, 1.0]
    elif args.stage == 4:
        train_config.iter_weights = [0.125, 0.25, 0.5, 1.0]
    else:
        print("stage must be in [1,2,3,4]")
    
    if GLOBAL_DROPOUT_RATE != None:
        encoder_config, gfn_config, train_config = \
            tuple([set_dropout_rate_config(c, GLOBAL_DROPOUT_RATE)
                        for c in [encoder_config, gfn_config, train_config]])
    
    modules = {
        "encoder": {"module": MolCT_Plus, 
                    "args": {"config": encoder_config}, 
                    "freeze": False},
        "gfn": {"module": GFNReadout, 
                "args": {"config": gfn_config},
                "freeze": False}
    }
    
    if args.load_ckpt_path:
        ##### load params
        with open(args.load_ckpt_path, "rb") as f:
            params = pkl.load(f)
            params = jax.tree_map(lambda x: jnp.array(x), params)
        
    ##### freeze params of encoder/decoder/vq_tokenizer
    for k, v in modules.items():
        if v["freeze"]:
            modules[k]['args']['config'] = \
                set_dropout_rate_config(modules[k]['args']['config'], 0.0)
        modules[k]["module"] = v["module"](**v["args"])
        if v["freeze"]:
            partial_params = {"params": params["params"]['genneg_net'].pop(k)}
            modules[k]["module"] = partial(modules[k]["module"].apply, partial_params)
    
    moledit_scorenet = MolEditScoreNet(
        encoder = modules['encoder']['module'], 
        gfn = modules['gfn']['module'],
    )
    moledit_ve_with_loss_cell = MolEditWithVELossCell(
        score_net = moledit_scorenet,
        train_cfg = train_config
    )
    
    if args.load_ckpt_path:
        ##### load params
        # with open(args.load_ckpt_path, "rb") as f:
        #     params = pkl.load(f)
        #     params = jax.tree_map(lambda x: jnp.array(x), params)
        pass
    else:
        name_list, select_bins = orgnize_name_list(name_list_bin, 
                                                   bins=DATA_BINS, 
                                                   batch_size=LOCAL_BATCH_SIZE,
                                                   num_batches=1,
                                                   p_scaling=np.ones(NUM_BINS, dtype=np.float32)) 
        init_data_dict = \
            load_train_data_pickle(name_list,
                                   start_idx=0,
                                   end_idx=2,
                                   num_parallel_worker=1,
                                   n_samples_per_structure=args.n_samples_per_structure, # 4,
                                   feature_processed=True,
                                   allowed_mask_types=args.allowed_mask_types)
        for k, v in init_data_dict.items():
            print("\t{}: {} {}".format(k, v[0].shape, v.dtype))
        init_data = [init_data_dict[k][0] for k in moledit_input_key_names]
        init_data = jax.tree_map(jnp.array, init_data)
        state_key, param_key, dropout_key, rng_key = jax.random.split(rng_key, 4)
        init_rngs = {'stats': state_key, 
                     'params': param_key, 
                     'dropout': dropout_key}
        params = moledit_ve_with_loss_cell.init(init_rngs, *init_data)
        with open(os.path.join(
            args.save_ckpt_dir, f"moledit_params_0.pkl"), "wb") as f:
            pkl.dump(
                jax.tree_map(lambda x: np.array(x), params), f
            )
            
    ##### initialize opetimizer 
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value = train_config.lr.lr_min,
        peak_value = train_config.lr.lr_max,
        warmup_steps = train_config.lr.warmup_steps,
        decay_steps = train_config.lr.lr_decay_steps, 
        end_value = train_config.lr.lr_min
    )
    optimizer = optax.chain(optax.clip_by_global_norm(args.gradient_clip),
                            optax.adam(learning_rate=lr_schedule))
    
    if args.load_opt_state_path:
        with open(args.load_opt_state_path, "rb") as f:
            opt_state = pkl.load(f)
            opt_state = jax.tree_map(lambda x: jnp.array(x), opt_state)
    else:
        opt_state = optimizer.init(params)
        
    ##### replicate opt_state
    params = replicate(params)
    opt_state = replicate(opt_state)
        
    ##### prepare train functions, jit & vmap
    with_loss_cell_jvj = jax.jit(
        jax.vmap(jax.jit(moledit_ve_with_loss_cell.apply), 
                 in_axes=[None] + [0] * len(moledit_input_key_names))
    )
    
    def forward(params, batch_input, net_rng_key):
        loss, loss_dict, peffective_atom_numbers = moledit_ve_forward_per_device(
            with_loss_cell_jvj, params, batch_input, net_rng_key)
        loss_dict["weight_decay"] = parameter_weight_decay(params) / float(NGLOBAL_DEVICES)
        loss = loss + args.weight_decay * loss_dict["weight_decay"]
        return loss, (loss_dict, peffective_atom_numbers)
    
    def loss_and_grad(params, batch_input, net_rng_key):
        loss, grad = jax.value_and_grad(forward, has_aux=True)(params, batch_input, net_rng_key)
        return loss, grad 

    def psum_tree(values):
        return jax.tree_map(
            lambda x: jax.lax.psum(x, axis_name="i"), values
        )
    
    def ema_value_update(value, update_value_mean, peffective_atom_numbers):
        value = value * ema_decay + \
            update_value_mean * jnp.power(peffective_atom_numbers, 
                                          1.0/train_config.atom_number_power)
        return value
    
    ### jitted functions
    pmap_jit_loss_and_grad = jax.pmap(jax.jit(loss_and_grad), axis_name="i")
    pmap_mean_scalar = jax.pmap(jax.jit(
        lambda x:jax.lax.psum(x, axis_name="i")), axis_name="i")
    pmap_mean_tree_aux_loss = jax.pmap(jax.jit(psum_tree), axis_name="i", in_axes=(loss_name_dict,))
    pmap_mean_tree_grads = jax.pmap(jax.jit(psum_tree), axis_name="i", in_axes=(jax.tree_map(lambda x:0, params),))
    pmap_jit_optimizer_update = jax.pmap(jax.jit(optimizer.update), axis_name="i")
    pmap_jit_apply_updates = jax.pmap(jax.jit(optax.apply_updates), axis_name="i")
    pmap_jit_ema_value_update = jax.pmap(jax.jit(ema_value_update), axis_name="i")
    
    #### dynamic bucket sampling algorithm 
    bin_losses, ema_decay = jax.tree_map(lambda x:jnp.array(x, dtype=jnp.float32), 
                                         {b: jnp.ones(NLOCAL_DEVICES, dtype=jnp.float32) for b in DATA_BINS}), 0.9
    
    if (RANK == 0):
        f = open(args.logger_path, "w")
        logger(f, "=====================START TRAINING=====================")
    
    for step in range(args.start_step, TOTAL_STEP):
        if (step % args.logger_step == 0):
            # timing
            start_time = datetime.datetime.now()

        #### load data
        PRE_LOAD_STEP = 20
        # if (step_ % PRE_LOAD_STEP == 0 or step == args.start_step):
        if (step % PRE_LOAD_STEP == 0 or step == args.start_step):
            start_time_load = datetime.datetime.now()
            start_data_idx = 0
            end_data_idx = start_data_idx + LOCAL_BATCH_SIZE * PRE_LOAD_STEP
            
            p_scaling = np.array([bin_losses[b][0] for b in DATA_BINS], dtype=np.float32)
            p_scaling = np.power(p_scaling, train_config.bucket_scaling_power) 
            # p_scaling = np.ones(NUM_BINS, dtype=np.float32)
            name_list, select_bins = \
                orgnize_name_list(name_list_bin, 
                                  bins=DATA_BINS, 
                                  batch_size=LOCAL_BATCH_SIZE,
                                  num_batches=PRE_LOAD_STEP,
                                  p_scaling=p_scaling) 
            
            train_dataset_dict = \
                load_train_data_pickle(name_list, 
                                       start_idx=start_data_idx, 
                                       end_idx=end_data_idx,
                                       num_parallel_worker=32,
                                       n_samples_per_structure=args.n_samples_per_structure, # 4,
                                       feature_processed=True,
                                       allowed_mask_types=args.allowed_mask_types)
            batch_input_pre_load = [train_dataset_dict[name] for name in moledit_input_key_names]
            end_time_load = datetime.datetime.now()
            data_trunk_id = 0
        
        batch_input = jax.tree_map(lambda x: jnp.array(x[data_trunk_id*LOCAL_BATCH_SIZE:(data_trunk_id + 1)*LOCAL_BATCH_SIZE]), batch_input_pre_load)
        current_bin = select_bins[data_trunk_id]
        data_trunk_id = data_trunk_id + 1
        
        #### split keys
        rng_keys = jax.random.split(rng_key, num=LOCAL_BATCH_SIZE + 1)
        net_rng_key = {"dropout": rng_keys[:-1]}
        rng_key = rng_keys[-1]
        
        #### reshape inputs 
        reshape_func = lambda x:x.reshape(NLOCAL_DEVICES, x.shape[0]//NLOCAL_DEVICES, *x.shape[1:])
        batch_input = jax.tree_map(reshape_func, batch_input)
        net_rng_key = jax.tree_map(reshape_func, net_rng_key)
        
        loss, grad = pmap_jit_loss_and_grad(params, batch_input, net_rng_key)
        loss, (aux_loss_dict, peffective_atom_numbers) = loss
        peffective_atom_numbers = peffective_atom_numbers / float(GLOBAL_BATCH_SIZE)
        
        ### ema update loss for dynamic bucket sampling 
        bin_losses[current_bin] = pmap_jit_ema_value_update(bin_losses[current_bin], 
                                                       loss, peffective_atom_numbers)
        
        # weighted mean
        loss = pmap_mean_scalar(loss)
        aux_loss_dict = pmap_mean_tree_aux_loss(aux_loss_dict)
        grad = pmap_mean_tree_grads(grad)
        
        updates, opt_state = pmap_jit_optimizer_update(grad, opt_state, params) ## Liyh at 12-20: remove jit outside
        params = pmap_jit_apply_updates(params, updates)
        
        if ((step + 1) % args.logger_step == 0 and RANK == 0):
            end_time = datetime.datetime.now()
            
            logger(f, "STEP: {}".format((step + 1)))
            logger(f, "DATA BIN: {}".format(current_bin))
            logger(f, "\tloss: {:.4f}".format(loss[0]))
            logger(f, "\tpeffective_atom_numbers: {:.4f}".format(peffective_atom_numbers[0]))
            for k, v in aux_loss_dict.items():
                logger(f, "\t{}: {:.4f}".format(k, v[0]))
            logger(f, "\tlikelihood per mol")
            for b, mol_loss in bin_losses.items():
                logger(f, "\t\t{}: {:.4f}".format(b, mol_loss[0]))
            logger(f, "\ttime: {} & {}".format(end_time - start_time, end_time_load - start_time_load), flush=True)
            
            if ((step + 1) % args.save_ckpt_step == 0):
                
                with open(os.path.join(args.save_ckpt_dir, 
                                        f"moledit_params_{step + 1}.pkl"), "wb") as f_ckpt:
                    pkl.dump(jax.tree_map(lambda x: np.array(x[0], dtype=np.float32), params), f_ckpt)
                with open(os.path.join(args.save_ckpt_dir, 
                                        f"opt_state_{step + 1}.pkl"), "wb") as f_ckpt:
                    pkl.dump(jax.tree_map(lambda x: np.array(x[0]), opt_state), f_ckpt)
                    
    if (RANK == 0):
        logger(f, "=====================END TRAINING=====================")        
        f.close()
    print("done")
    
if __name__ == "__main__":
    train()