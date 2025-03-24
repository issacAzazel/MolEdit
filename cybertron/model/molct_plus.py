"""MolCT Plus by Justin Zhang."""

import jax
import jax.numpy as jnp
import flax.linen as nn

from ..common.config_load import Config
from ..common.base import FeatureTransformer, str_to_jax_dtype
from .interaction.molct_interaction import TopoInteractionUnit, AdaLNTopoInteractionUnit

from config.global_setup import EnvironConfig
global_setup = EnvironConfig() ### Set Hyper-parameters here
NORM_SMALL = global_setup.norm_small
BF16_FLAG = global_setup.bf16_flag

class MolCT_Plus(nn.Module):

    config: Config

    @nn.compact
    def __call__(self, atom_raw_feat, pair_raw_feat, atom_mask):
        r"""MolCT+ by ZhangJ.
        Args:
            atom_raw_feat: [A, Cin];
            pair_raw_feat: [A, A, Cin'];
            atom_mask: [A,];
        Return:
            mol_act: [Cm,]
            atom_feat: [A, Cm]
            bond_feat: [A, A, Cz]
        """
        
        _dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32

        config = self.config.model
        settings = self.config.settings
        niu_num_block = settings.niu_num_block
        epsilon = settings.epsilon

        ### Feature Generator:
        ft_config = config.feat_generator
        atom_types = ft_config.atom_types
        pair_types = ft_config.pair_types
        atom_act_dim = ft_config.atom_act_dim
        pair_act_dim = ft_config.pair_act_dim
        use_cls = ft_config.use_cls
        fp_type = str_to_jax_dtype(ft_config.fp_type)
        
        ##### Warning: fp_type is abandoned, using global config instead
        
        _feat_gen_blocks = []
        for i in range(niu_num_block):
            _block = FeatureTransformer(atom_types, pair_types, atom_act_dim, pair_act_dim, use_cls, fp_type)
            _feat_gen_blocks.append(_block)
        feat_generator = _feat_gen_blocks

        tiu_config = config.interaction_unit
        ### Encoder:
        _encoder_blocks = []
        for i in range(niu_num_block):
            _niu_block = TopoInteractionUnit(tiu_config)
            _encoder_blocks.append(_niu_block)
        encoder = _encoder_blocks

        # ### Helper ops:
        # self.matmul_trans_b = P.MatMul(transpose_b=True)
        # self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)

        ### 0. Compute MaskNorm for OuterProduct:
        # (A, 1) @ (1, A) -> (A, A)
        bond_mask = jnp.logical_and(jnp.expand_dims(atom_mask, -1), jnp.expand_dims(atom_mask, -2))
        # -> (A, A, 1)
        mask_norm = jnp.expand_dims(bond_mask, -1)

        ### 1. Execute Encoder:
        atom_feat = 0
        bond_feat = 0
        for i in range(niu_num_block):
            ## 1) Re-initialize Features:
            atom_feat_resample, bond_feat_resample = feat_generator[i](atom_raw_feat, pair_raw_feat)
            atom_feat += atom_feat_resample
            bond_feat += bond_feat_resample
            
            ## 2) Run NIU:
            # (A, Cm), (A, A, Cz):
            atom_feat, bond_feat = encoder[i](atom_feat, bond_feat, atom_mask, bond_mask)
        
        # (A, 1)
        tmp_mask = jnp.expand_dims(atom_mask, -1)
        # (Cm,):
        mol_act = jnp.sum(atom_feat * tmp_mask, -2) / (jnp.sum(atom_mask) + epsilon) ### bug here

        # @ZhangJ. Add QFormer for pooling:
        
        if use_cls:
            mol_act = atom_feat[0, :]

        return mol_act, atom_feat, bond_feat
    
class AdaLNMolCT_Plus(nn.Module):

    config: Config

    @nn.compact
    def __call__(self, atom_raw_feat, pair_raw_feat, atom_mask, cond):
        r"""MolCT+ by ZhangJ.
        Args:
            atom_raw_feat: [A, Cin];
            pair_raw_feat: [A, A, Cin'];
            atom_mask: [A,];
        Return:
            mol_act: [Cm,]
            atom_feat: [A, Cm]
            bond_feat: [A, A, Cz]
        """
        
        _dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32

        config = self.config.model
        settings = self.config.settings
        niu_num_block = settings.niu_num_block
        epsilon = settings.epsilon

        ### Feature Generator:
        ft_config = config.feat_generator
        atom_types = ft_config.atom_types
        pair_types = ft_config.pair_types
        atom_act_dim = ft_config.atom_act_dim
        pair_act_dim = ft_config.pair_act_dim
        use_cls = ft_config.use_cls
        fp_type = str_to_jax_dtype(ft_config.fp_type)
        
        ##### Warning: fp_type is abandoned, using global config instead
        
        _feat_gen_blocks = []
        for i in range(niu_num_block):
            _block = FeatureTransformer(atom_types, pair_types, atom_act_dim, pair_act_dim, use_cls, fp_type)
            _feat_gen_blocks.append(_block)
        feat_generator = _feat_gen_blocks

        tiu_config = config.interaction_unit
        ### Encoder:
        _encoder_blocks = []
        for i in range(niu_num_block):
            _niu_block = AdaLNTopoInteractionUnit(tiu_config)
            _encoder_blocks.append(_niu_block)
        encoder = _encoder_blocks

        # ### Helper ops:
        # self.matmul_trans_b = P.MatMul(transpose_b=True)
        # self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)

        ### 0. Compute MaskNorm for OuterProduct:
        # (A, 1) @ (1, A) -> (A, A)
        bond_mask = jnp.logical_and(jnp.expand_dims(atom_mask, -1), jnp.expand_dims(atom_mask, -2))
        # -> (A, A, 1)
        mask_norm = jnp.expand_dims(bond_mask, -1)

        ### 1. Execute Encoder:
        atom_feat = 0
        bond_feat = 0
        for i in range(niu_num_block):
            ## 1) Re-initialize Features:
            atom_feat_resample, bond_feat_resample = feat_generator[i](atom_raw_feat, pair_raw_feat)
            atom_feat += atom_feat_resample
            bond_feat += bond_feat_resample
            
            ## 2) Run NIU:
            # (A, Cm), (A, A, Cz):
            atom_feat, bond_feat = encoder[i](atom_feat, bond_feat, atom_mask, bond_mask, cond)
        
        # (A, 1)
        tmp_mask = jnp.expand_dims(atom_mask, -1)
        # (Cm,):
        mol_act = jnp.sum(atom_feat * tmp_mask, -2) / (jnp.sum(atom_mask) + epsilon) ### bug here

        # @ZhangJ. Add QFormer for pooling:
        
        if use_cls:
            mol_act = atom_feat[0, :]

        return mol_act, atom_feat, bond_feat

### MolCT Plus Without FeatureTransformer    
class MolCT_Interaction(nn.Module):

    config: Config
    num_layer: int = 1
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, atom_feat, bond_feat, atom_mask):
        r"""MolCT+ by ZhangJ.
        Args:
            atom_feat: [B,A,C];
            bond_feat: [B,A,A,C'];
            atom_mask: [B,A];
        Return:
            mol_act: [Cm,]
            atom_feat: [A, Cm]
            bond_feat: [A, A, Cz]   
        """
        
        config = self.config.model
        flag_use_cls = config.feat_generator.use_cls

        ### Encoder:
        tiu_config = config.topo_interaction_unit
        _encoder_blocks = []
        for i in range(self.num_layer):
            _niu_block = TopoInteractionUnit(tiu_config)
            _encoder_blocks.append(_niu_block)
        encoder = _encoder_blocks

        # ### Helper ops:
        # self.matmul_trans_b = P.MatMul(transpose_b=True)
        # self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)

        ### 0. Compute MaskNorm for OuterProduct:
        # (A, 1): 
        mask_tmp = jnp.expand_dims(atom_mask, -1)
        # (A, 1) @ (1, A) -> (A, A)
        bond_mask = jnp.logical_and(mask_tmp, mask_tmp.T)
        # -> (A, A, 1)
        mask_norm = jnp.expand_dims(bond_mask, -1)

        ### 1. Execute Encoder:
        for i in range(self.num_layer):            
            ## 1) Run NIU:
            # (A, Cm), (A, A, Cz):
            atom_feat, bond_feat = encoder[i](atom_feat,bond_feat,atom_mask,bond_mask,mask_norm)
        
        # (C,):
        mol_act = jnp.sum(atom_feat * mask_tmp, -2) / (jnp.sum(mask_tmp, -2) + self.epsilon)
        if flag_use_cls:
            mol_act = atom_feat[0,:] 

        return mol_act, atom_feat, bond_feat