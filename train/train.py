import numpy as np 
import jax 
import jax.numpy as jnp 
import flax.linen as nn
from typing import Union, Callable

from cybertron.common.config_load import Config
from cybertron.model.molct_plus import MolCT_Plus
from cybertron.readout import GFNReadout, GFNScalarReadout

from config.global_setup import EnvironConfig
global_setup = EnvironConfig() ### Set Hyper-parameters here
NORM_SMALL = global_setup.norm_small
BF16_FLAG = global_setup.bf16_flag

class MolEditScoreNet(nn.Module):
    encoder: Union[nn.Module, Callable]
    gfn: Union[nn.Module, Callable]
    with_cond: bool = False
    
    @nn.compact
    def __call__(self, atom_raw_feat, pair_raw_feat, xi, atom_mask, noise, rg, cond=None):
        if self.with_cond:
            _, atom_feat, bond_feat = self.encoder(atom_raw_feat, pair_raw_feat, atom_mask, cond)
            displacements = self.gfn(atom_feat, bond_feat, xi, atom_mask, noise, rg, cond)
        else:
            _, atom_feat, bond_feat = self.encoder(atom_raw_feat, pair_raw_feat, atom_mask)
            displacements = self.gfn(atom_feat, bond_feat, xi, atom_mask, noise, rg)

        return displacements
    
class MolEditWithVELossCell(nn.Module):
    score_net: nn.Module
    train_cfg: Config
    eps: float = 1e-6
    
    def setup(self):
        self.iter_weights = jnp.array(self.train_cfg.iter_weights)
        self.iter_weights = self.iter_weights / jnp.sum(self.iter_weights)
        self.atom_number_power = self.train_cfg.atom_number_power

    def __call__(self, atom_feat, bond_feat, xi, atom_mask, noise, label, rg, cond=None):
        
        ##### convert to _dtype
        _dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        atom_feat, bond_feat = \
            jax.tree_map(lambda x:x.astype(_dtype), 
                        (atom_feat, bond_feat))
        displacements = self.score_net(atom_feat, bond_feat, xi, 
                                            atom_mask, noise, rg, cond)
        displacements = jnp.array(displacements)
        
        ##### convert to fp32
        displacements, label, noise, atom_mask, rg = jax.tree_map(
            lambda x: jnp.array(x, dtype=jnp.float32), (displacements, label, noise, atom_mask, rg)
        )
        
        normalized_label = label / (noise + self.eps) # (NATOM, 3)
        natom = jnp.sum(atom_mask)
        
        mse_traj = jnp.sum(
            (displacements - jnp.expand_dims(normalized_label, 0))**2, axis=-1) # (NITER, NATOM)
        mse_traj = jnp.sum(mse_traj * jnp.expand_dims(atom_mask, 0), -1) / \
                        (natom + self.eps)
                        
        mse = jnp.sum(mse_traj * self.iter_weights)
        
        loss_dict = {
            "loss": mse,
            "mse_last_iter": mse_traj[-1],
        }
        atom_number_weight = jnp.power(natom, self.atom_number_power)
        
        return loss_dict, atom_number_weight
    
def moledit_ve_forward(vmap_fn, params, batch_input, net_rng_key):
    loss_dict, atom_number_weight = vmap_fn(params, *batch_input, rngs=net_rng_key)
    
    atom_number_weight = atom_number_weight /(jnp.sum(atom_number_weight) + 1e-6)
    loss_dict = jax.tree_map(lambda x: jnp.sum(x * atom_number_weight), loss_dict)
    
    loss = loss_dict.pop("loss")
    return loss, loss_dict

def moledit_ve_forward_per_device(vmap_fn, params, batch_input, net_rng_key):
    loss_dict, atom_number_weight = vmap_fn(params, *batch_input, rngs=net_rng_key)

    effective_atom_numbers = \
        jax.lax.psum(jnp.sum(atom_number_weight), axis_name="i") + 1e-6
    atom_number_weight = atom_number_weight / effective_atom_numbers
        
    loss_dict = jax.tree_map(lambda x: jnp.sum(x * atom_number_weight), loss_dict)
    loss = loss_dict.pop("loss")
    
    return loss, loss_dict, effective_atom_numbers

class MolEditGenneGNet(nn.Module):
    encoder: Union[nn.Module, Callable]
    gfn: Union[nn.Module, Callable]
    
    def __call__(self, atom_raw_feat, pair_raw_feat, xi, atom_mask, noise, rg):
        _, atom_feat, bond_feat = self.encoder(atom_raw_feat, pair_raw_feat, atom_mask)
        logits = self.gfn(atom_feat, bond_feat, xi, atom_mask, noise, rg)

        return logits

class MolEditWithGenneGLossCell(nn.Module):
    genneg_net: nn.Module
    train_cfg: Config
    balance_ratio: float 
    eps: float = 1e-6
    
    def setup(self):
        self.atom_number_power = self.train_cfg.atom_number_power

    def __call__(self, atom_feat, bond_feat, xi, atom_mask, noise, rg, label):
        logits = self.genneg_net(atom_feat, bond_feat, xi, atom_mask, noise, rg)
        label = label.astype(jnp.float32)
        
        ##### debug 
        # logits = -logits
        prob = jax.nn.sigmoid(logits)
        
        loss = self.balance_ratio * label * (-nn.log_sigmoid(logits)) + \
                (1.0 - self.balance_ratio) * (1.0 - label) * (-nn.log_sigmoid(-logits))

        natom = jnp.sum(atom_mask.astype(jnp.float32))

        pos_prob = prob * label 
        neg_prob = (1.0 - prob) * (1.0 - label)
                        
        loss_dict = {
            "loss": loss,
            "prob": pos_prob + neg_prob,
            "pos_prob": pos_prob * 2.0, #### pos:neg = 1:1
            "neg_prob": neg_prob * 2.0, #### pos:neg = 1:1
            # prob * label + (1.0 - label) * (1.0 - prob),
            # "label": label,
        }
        atom_number_weight = jnp.power(natom, self.atom_number_power)
        
        return loss_dict, atom_number_weight
    
def moledit_genneg_forward(vmap_fn, params, batch_input, net_rng_key):
    loss_dict, atom_number_weight = vmap_fn(params, *batch_input, rngs=net_rng_key)
    
    atom_number_weight = atom_number_weight /(jnp.sum(atom_number_weight) + 1e-6)
    loss_dict = jax.tree_map(lambda x: jnp.sum(x * atom_number_weight), loss_dict)
    
    loss = loss_dict.pop("loss")
    
    ### debug
    # loss = jnp.sum(loss * atom_number_weight)
    return loss, loss_dict

def moledit_genneg_forward_per_device(vmap_fn, params, batch_input, net_rng_key):
    loss_dict, atom_number_weight = vmap_fn(params, *batch_input, rngs=net_rng_key)

    atom_number_weight = atom_number_weight / \
        (jax.lax.psum(jnp.sum(atom_number_weight), axis_name="i") + 1e-6)
        
    loss_dict = jax.tree_map(lambda x: jnp.sum(x * atom_number_weight), loss_dict)
    loss = loss_dict.pop("loss")
    
    return loss, loss_dict