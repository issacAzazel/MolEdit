# GFN Readout

import jax
import math
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import flax

from jax import lax
import e3nn_jax as e3nn
from flax.linen.initializers import lecun_normal, zeros_init
from ..common.config_load import Config
from ..common.layers.dense import Dense, LoRAModulatedDense
from ..common.layers.mlp import MLP, LoRAModulatedMLP
from ..common.cutoff import get_cutoff, GaussianCutoff, NormalizedGaussianCutoff, CosineCutoff
from ..common.rbf import LogGaussianBasis, BesselBasis
from ..common.activation import get_activation

from .readout import Readout, _readout_register
from .utils import MLPEdgeDecoder, MLPEdgeEncoder, LogSinusoidalNoiseEmb, LogGaussianBasisWithFlexRmax, BesselBasisWithFlexRmax, FourierBasis
from ..model.interaction.schnet_interaction import SchnetInteraction, HyperSchnetInteraction, LoRAModulatedSchnetInteraction
from ..model.allegro import Allegro, LoRAModulatedAllegro
from ..common.base import LoRAModulatedHyperformerPairBlock
from ..common.adaptive_layernorm import adaLN

from cybertron.modules.basic import ActFuncWrapper
from typing import Optional, Union, Tuple, List

from config.global_setup import EnvironConfig
global_setup = EnvironConfig() ### Set Hyper-parameters here
NORM_SMALL = global_setup.norm_small
BF16_FLAG = global_setup.bf16_flag
SAFE_PRECISION_FLAG = global_setup.safe_precision_flag
REMAT_FLAG = global_setup.remat_flag

####### for allegro
def create_edge_index_and_mask(pair_rep, pair_mask):

    n_atom, n_atom, dim_z = pair_rep.shape
    eye_mask = jnp.logical_not(jnp.eye(n_atom, dtype=jnp.bool_))
    edge_mask = (jnp.logical_and(eye_mask, pair_mask.astype(jnp.bool_))).reshape(-1)
    ns_ = jnp.arange(n_atom, dtype=jnp.int32)
    senders = jnp.expand_dims(ns_, axis=0).repeat(n_atom, axis=0).reshape(-1)
    receviers = jnp.expand_dims(ns_, axis=1).repeat(n_atom, axis=1).reshape(-1)
    edge_index = jnp.stack([senders, receviers], axis=0)

    return edge_index, edge_mask    
    
class AdaLNLoRAModulatedGFNIteration(nn.Module):

    config: Config ## Liyh: need to be implemented
    predict_scalar_logit: bool = False
    predict_vector_pos: bool = True
    edge_encoding: bool = True
    edge_decoding: bool = True

    def setup(self):
        
        self._dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        self._safedtype = jnp.float32 if SAFE_PRECISION_FLAG else self._dtype
        
        config = self.config
        
        ##### settings
        self.short_range_rmax_min = config.settings.short_range_rmax_min
        self.short_range_rmax_intercept = config.settings.short_range_rmax_intercept
        self.short_range_rmax_coef = config.settings.short_range_rmax_coef
        self.long_range_scale_factor = config.settings.long_range_scale_factor
        
        self.mixed_rbf = config.settings.mixed_rbf
        self.n_node_interactions = config.settings.n_node_interactions
        
        model_config = config.model
        ##### LoRA
        self.lora_rank = model_config.lora_rank
        self.lora_dropout_rate = model_config.lora_dropout_rate
        
        ###### rbfs
        self.n_log_gaussian_rbf = model_config.num_log_gaussian_rbfs
        if self.mixed_rbf:
            self.n_short_range_rbf = model_config.num_short_range_rbfs
        self.n_rbf = (self.n_log_gaussian_rbf + self.n_short_range_rbf) if self.mixed_rbf else self.n_log_gaussian_rbf
        self.activation = get_activation(model_config.activation)

        self.cutoff_fn = NormalizedGaussianCutoff(cutoff=None, sigma=None) 
        self.log_gaussian_rbf = LogGaussianBasisWithFlexRmax(
                                    r_min=model_config.log_gaussian_rmin,                          
                                    num_basis=self.n_log_gaussian_rbf,
                                    dtype=self._safedtype)
        
        self.short_range_rbf = model_config.short_range_rbf
        assert self.short_range_rbf in ['bessel', 'fourier'], 'unsupported short range rbf {}'.format(self.short_range_rbf)
        if self.short_range_rbf == 'bessel':
            self.bessel_rbf = BesselBasisWithFlexRmax(
                                        num_basis=self.n_short_range_rbf,
                                        trainable=model_config.bessel_trainable,
                                        dtype=self._safedtype)
        else:
            self.fourier_rbf = FourierBasis(num_basis=self.n_short_range_rbf,
                                            clamp_input=True, 
                                            scale='log',
                                            decay='right',
                                            dtype=self._safedtype)
            
        self.rbf_dense = LoRAModulatedDense(dim_out=self.n_rbf,         
                                            activation=self.activation,
                                            lora_rank=self.lora_rank,
                                            lora_dropout_rate=self.lora_dropout_rate,
                                            dtype=self._safedtype)

        ##### dimension of representations
        self.dim_s = model_config.dim_s 
        self.dim_z = model_config.dim_z 
        self.dim_e = model_config.dim_e

        ##### noise & Rg embedding
        noise_emb_config = model_config.NoiseEmbedding
        self.dim_noise_emb = noise_emb_config.dim_emb
        self.noise_embedding = FourierBasis(num_basis=self.dim_noise_emb,
                                            clamp_input=True, 
                                            scale='log',
                                            decay=None,
                                            dtype=self._safedtype)
        
        self.transition_noise_scale = noise_emb_config.transition_noise_scale
        self.noise_clamp_min = noise_emb_config.noise_clamp_min / self.transition_noise_scale
        self.noise_clamp_max = noise_emb_config.noise_clamp_max / self.transition_noise_scale
        self.noise_linear_1 = nn.Dense(self.dim_noise_emb, use_bias=False,
                                       dtype=self._safedtype, param_dtype=jnp.float32)
        self.noise_linear_2 = nn.Dense(self.dim_noise_emb, use_bias=False,
                                       dtype=self._safedtype, param_dtype=jnp.float32)
        
        ##### shape embedding 
        shape_encoding_config = model_config.ShapeEncoding
        self.shape_length_scales = jnp.array(shape_encoding_config.length_scales, dtype=self._safedtype)
        self.dim_shape_emb = shape_encoding_config.dim_emb
        self.shape_embedding = FourierBasis(num_basis=self.dim_shape_emb,
                                            clamp_input=True, 
                                            scale='normal',
                                            decay=None, 
                                            dtype=self._safedtype)
        self.shape_linear = nn.Dense(self.dim_noise_emb, use_bias=False, 
                                     dtype=self._safedtype, param_dtype=jnp.float32)

        ##### atom encoder
        schnet_config = model_config.SchnetInteraction
        SchNetUnit = nn.checkpoint(LoRAModulatedSchnetInteraction) if REMAT_FLAG \
                        else LoRAModulatedSchnetInteraction
        self.atom_encoder_cells = \
            [
                adaLN(SchNetUnit(
                    dim_filter = schnet_config.dim_filter,
                    n_filter_hidden = schnet_config.n_filter_hidden,
                    activation = schnet_config.activation,
                    filter_activation = schnet_config.filter_activation,
                    normalize_filter = schnet_config.normalize_filter,
                    dropout_rate = schnet_config.dropout_rate,
                    lora_rank = self.lora_rank,
                    lora_dropout_rate = self.lora_dropout_rate, 
                    residual=False
                )) for _ in range(self.n_node_interactions)
            ]

        self.skip_connection_rbf_projection_in_cells = \
            [
                LoRAModulatedDense(
                    self.dim_z, activation=self.activation,
                    lora_rank=self.lora_rank,
                    lora_dropout_rate=self.lora_dropout_rate) 
                        for _ in range(self.n_node_interactions)
            ]
        self.skip_connection_rbf_projection_out_cells = \
            [
                LoRAModulatedDense(
                    self.dim_z, activation=self.activation,
                    lora_rank=self.lora_rank,
                    lora_dropout_rate=self.lora_dropout_rate) 
                        for _ in range(self.n_node_interactions)
            ]
        
        ##### edge encoder
        if self.edge_encoding:
            self.edge_encoder_mode = model_config.edge_encoder_mode
            assert self.edge_encoder_mode in ['mlp', 'allegro', 'outerproduct'], 'unsupported edge encoder!'
            if self.edge_encoder_mode == 'mlp':
                encoder_config = model_config.MLPEdgeEncoder
                self.edge_encoder = MLPEdgeEncoder(dim_s=self.dim_s, 
                                                dim_z=self.dim_z, 
                                                dim_rbf=self.n_rbf, 
                                                dim_edge=self.dim_e, 
                                                layer_dims=encoder_config.layer_dims, 
                                                activation=encoder_config.activation)
            elif self.edge_encoder_mode == 'allegro':
                encoder_config = model_config.Allegro 
                self.lora_allegro = encoder_config.lora
                irrep_str = ""
                for i in range(encoder_config.max_ell):
                    irrep_str += f"{i}e + {i}o + "
                irrep_str += f"{encoder_config.max_ell}e + {encoder_config.max_ell}o"

                allegro_module = LoRAModulatedAllegro if self.lora_allegro \
                                    else Allegro
                allegro_module = nn.checkpoint(allegro_module) if REMAT_FLAG else allegro_module
                
                allegro_args = {
                    "avg_num_neighbors": model_config.dim_z, 
                    "max_ell": encoder_config.max_ell, 
                    "irreps": e3nn.Irreps(irrep_str),
                    "mlp_activation": get_activation(encoder_config.mlp_activation),
                    "mlp_n_hidden": encoder_config.mlp_n_hidden,
                    "mlp_n_layers": encoder_config.mlp_n_layers,
                    "output_irreps": e3nn.Irreps("0e"), 
                    "num_layers": encoder_config.num_layers,
                    "layer_norm_reps": encoder_config.layer_norm_reps,
                    "env_n_channel": encoder_config.env_n_channel,
                    "output_n_channel": self.dim_z,
                    "eps": encoder_config.eps,
                    "shared_weights_flag": encoder_config.share_weights,
                    "gradient_normalization": encoder_config.gradient_normalization
                }
                if self.lora_allegro:
                    allegro_args.update(
                        {
                            "dropout_rate" : encoder_config.dropout_rate,
                            "lora_rank" : self.lora_rank,
                            "lora_dropout_rate" : self.lora_dropout_rate
                        }
                    )
                self.edge_encoder = allegro_module(**allegro_args)
            else:
                encoder_config = model_config.OuterProductPairBlock
                self.edge_encoder_layers = encoder_config.num_layers
                edge_encoders = []
                edge_rbf_projection_in_cells = []
                edge_rbf_projection_out_cells = []

                for i in range(self.edge_encoder_layers):
                    edge_encoder = LoRAModulatedHyperformerPairBlock(
                        dim_feature = self.dim_z, 
                        dim_outerproduct = encoder_config.dim_outer_pdct,
                        num_transition = encoder_config.num_transition,
                        act_fn = encoder_config.act_fn,
                        lora_rank = self.lora_rank,
                        lora_dropout_rate = self.lora_dropout_rate,
                    )
                    edge_encoders.append(edge_encoder)

                    edge_rbf_projection_in_cell = \
                            LoRAModulatedDense(
                                self.dim_z, activation=self.activation,
                                lora_rank=self.lora_rank,
                                lora_dropout_rate=self.lora_dropout_rate)
                    edge_rbf_projection_out_cell = \
                            LoRAModulatedDense(
                                self.dim_z, activation=self.activation,
                                lora_rank=self.lora_rank, 
                                lora_dropout_rate=self.lora_dropout_rate
                            ) 
                    edge_rbf_projection_in_cells.append(edge_rbf_projection_in_cell)
                    edge_rbf_projection_out_cells.append(edge_rbf_projection_out_cell)
                    
                self.edge_layer_norms = [ActFuncWrapper(nn.LayerNorm(name=f"layer_norm_z{_}",
                                                                     epsilon=NORM_SMALL, 
                                                                     dtype=self._dtype, 
                                                                     param_dtype=jnp.float32)) for _ in range(self.n_node_interactions)]
                    
                self.edge_encoders = edge_encoders
                self.edge_rbf_projection_in_cells = edge_rbf_projection_in_cells
                self.edge_rbf_projection_out_cells = edge_rbf_projection_out_cells
        
        ###### edge decoder
        if self.edge_decoding:
            decoder_config = model_config.MLPEdgeDecoder
            self.edge_decoder = LoRAModulatedMLP( # LoRAModulatedMLPDebug(
                                output_sizes = decoder_config.layer_dims + [1,],
                                activation = decoder_config.activation, 
                                dropout_rate = decoder_config.dropout_rate, 
                                activate_final = False,
                                lora_rank = self.lora_rank, 
                                lora_dropout_rate = self.lora_dropout_rate)
            
        ####### mol act decoder
        if self.predict_scalar_logit:
            decoder_config = model_config.MLPMolDecoder
            self.pooling = decoder_config.pooling 
            assert self.pooling in ['mean', 'max'], \
                        'pooling method {} is not supported'.format(self.pooling)
            self.layer_norm_mol = ActFuncWrapper(nn.LayerNorm(epsilon=NORM_SMALL, dtype=self._dtype, param_dtype=jnp.float32))
            self.mol_decoder = LoRAModulatedMLP( # LoRAModulatedMLPDebug(
                                output_sizes = decoder_config.layer_dims + [2,],
                                activation = decoder_config.activation, 
                                dropout_rate = decoder_config.dropout_rate, 
                                activate_final = False,
                                lora_rank = self.lora_rank, 
                                lora_dropout_rate = self.lora_dropout_rate)
            
        
        ###### layer norms
        self.layer_norm_noise = ActFuncWrapper(nn.LayerNorm(epsilon=NORM_SMALL, dtype=self._dtype, param_dtype=jnp.float32))
        self.layer_norm_z = ActFuncWrapper(nn.LayerNorm(epsilon=NORM_SMALL, dtype=self._dtype, param_dtype=jnp.float32, ))
        self.layer_norm_s_ = ActFuncWrapper(nn.LayerNorm(epsilon=NORM_SMALL, dtype=self._dtype, param_dtype=jnp.float32, ))
        self.layer_norm_s = [ActFuncWrapper(nn.LayerNorm(epsilon=NORM_SMALL, dtype=self._dtype, param_dtype=jnp.float32,)) for _ in range(self.n_node_interactions)]
        
        if self.edge_encoding:
            self.layer_norm_e = ActFuncWrapper(nn.LayerNorm(epsilon=NORM_SMALL, dtype=self._dtype, param_dtype=jnp.float32, ))
        
        ###### masks
        self.self_mask = 1.0 - jnp.eye(model_config.num_atoms, dtype=self._dtype)
    
    def preprocess_structural_features(self, x_i, atom_mask, mask_2d, sigma, rg):
        ###### Warning
        ###### This part of calculation should be under safe dtype
    
        x_i, sigma, rg = jax.tree_map(lambda x:jnp.array(x, dtype=self._safedtype), 
                                      (x_i, sigma, rg))
        
        d_ij = jnp.expand_dims(x_i, -2) - jnp.expand_dims(x_i, -3) # (A, 1, 3) - (1, A, 3)
        r_ij = (jnp.linalg.norm(
            d_ij.astype(jnp.float32) + (1.0 - mask_2d.astype(jnp.float32))[...,None] * NORM_SMALL, axis=-1)).astype(self._safedtype) ### to prevent nan bug
        
        short_range_rmax = jnp.maximum(self.short_range_rmax_intercept + sigma * self.short_range_rmax_coef,
                                       self.short_range_rmax_min)
        long_range_rmax = short_range_rmax * self.long_range_scale_factor
        cutoff_length = long_range_rmax
        c_ij, mask_2d = self.cutoff_fn(r_ij, mask_2d, 
                                       cutoff=cutoff_length) ##### scale here
        
        # get rbf: (Nl, A, A, Cz1) & (Nl, A, A, Cz2) -> (Nl, A, A, Cz1 + Cz2)
        f_ij = self.log_gaussian_rbf(r_ij, r_max=long_range_rmax) * c_ij[..., None]
        if self.mixed_rbf:
            if self.short_range_rbf == 'bessel':
                short_range_rbfs = self.bessel_rbf(r_ij, r_max=short_range_rmax)
            else:
                short_range_rbfs = self.fourier_rbf(r_ij, x_max=short_range_rmax, x_min=0.1)
            
            f_ij = jnp.concatenate([f_ij, short_range_rbfs], axis=-1)
        
        noise_emb = self.noise_embedding(sigma / self.transition_noise_scale, 
                                         x_max=self.noise_clamp_max, 
                                         x_min=self.noise_clamp_min)
        noise_emb = self.noise_linear_1(noise_emb)
        
        n_atoms = jnp.sum(atom_mask.astype(self._safedtype))
        shape_emb = (jnp.log(rg) - jnp.log(self.shape_length_scales)) / jnp.log(n_atoms)
        shape_emb = self.shape_embedding(shape_emb, 
                                         x_max=1.0, x_min=0.0).reshape(-1)
        
        noise_emb = self.noise_linear_2(self.shape_linear(shape_emb) + noise_emb)
        noise_emb = self.layer_norm_noise(noise_emb)
        
        ##### rbfs * decay factor
        f_ij = self.rbf_dense(f_ij, noise_emb)
        
        #### covert to dtype
        f_ij, noise_emb = \
            jax.tree_map(lambda x:jnp.array(x, dtype=self._dtype), 
                         (f_ij, noise_emb))
        return x_i, sigma, d_ij, r_ij, f_ij, c_ij, noise_emb, mask_2d
    
    def __call__(self, s_i, z_ij, x_i, s_mask, sigma, rg, cond):
        """
        ## Args:
            s_i (jax.Array): Shape of (B, A, Cs), single representation, a.k.a. node_rep

            z_ij (jax.Array): Shape of (B, A, A, Cz), edge representation, a.k.a. edge_rep

            x_i (jax.Array): Shape of (B, A, 3), coordinate.

            s_mask (jax.Array): Shape of (B, A), single representation mask.

            sigma (jax.Array): Shape of (B, Nrnf).

        """
        # s_i: (A, Cs), z_ij: (A, A, Cz), x_i: (A, 3), s_mask: (A,), sigma: ()
        
        # symmetrize z_ij
        z_ij = 0.5 * (z_ij + jnp.transpose(z_ij, (1, 0, 2)))
        # make mask
        s_mask_2d = jnp.expand_dims(s_mask, -1) * jnp.expand_dims(s_mask, -2) # (A, 1) * (1, A) -> (A, A)
        mask = self.self_mask * s_mask_2d # (A, A) * (A, A) -> (A, A)
        mask = jnp.asarray(mask, jnp.bool_)

        x_i, sigma, d_ij, r_ij, f_ij, c_ij, noise_emb, mask = \
            self.preprocess_structural_features(x_i, s_mask, mask, sigma, rg)
        
        s_i = self.layer_norm_s_(s_i) # normalize node rep
        z_ij = self.layer_norm_z(z_ij) # normalize edge rep
        
        # assert f_ij.dtype == jnp.bfloat16, 'fij dtype error 1'
        # assert c_ij.dtype == jnp.bfloat16, 'cij dtype error 1'
        # assert s_i.dtype == jnp.bfloat16, 'si dtype error 1'
        # assert z_ij.dtype == jnp.bfloat16, 'zij dtype error 1'
        # assert noise_emb.dtype == jnp.bfloat16, 'noise_emb dtype error 1'
        
        for k in range(self.n_node_interactions):
            # s_i = self.layer_norm_s[k](s_i) # pre-layernorm: normalize node rep
            s_i = self.atom_encoder_cells[k](
                s_i, 
                cond, 
                (self.skip_connection_rbf_projection_out_cells[k](
                    z_ij + self.skip_connection_rbf_projection_in_cells[k](f_ij, noise_emb), 
                    noise_emb), 
                c_ij.astype(self._dtype),
                mask,
                noise_emb,)
            )
            # s_i = self.layer_norm_s[k](s_i) # post_layer_norm: normalize node rep ### do we need post-LN in adaLN ver. of shnet?
            
        # assert s_i.dtype == jnp.bfloat16, 'si dtype error 2'
        
        if self.edge_encoding:
            # update edge
            if self.edge_encoder_mode == 'allegro':
                natom = s_i.shape[0]
                nedges = natom * natom
                edge_index, edge_mask = create_edge_index_and_mask(z_ij, mask)
                allegro_input = (
                    s_i, 
                    jnp.reshape(z_ij, (nedges, -1)), 
                    jnp.reshape(d_ij.astype(self._dtype), (nedges, -1)), 
                    jnp.reshape(r_ij.astype(self._dtype), (nedges, )), 
                    jnp.reshape(c_ij.astype(self._dtype), (nedges, )), 
                    edge_mask, edge_index, s_mask
                )
                if self.lora_allegro:
                    allegro_input += (noise_emb,)
                z_ij = self.edge_encoder(*allegro_input).array
            elif self.edge_encoder_mode == 'mlp':
                z_ij = self.edge_encoder(s_i, z_ij, f_ij) ## mlp
            else:
                #### outer product
                for i in range(self.edge_encoder_layers):
                    # assert s_i.dtype == jnp.bfloat16, f'{i} s_i dtype error inside'
                    # assert z_ij.dtype == jnp.bfloat16, f'{i} z_ij dtype error inside'
                    # assert f_ij.dtype == jnp.bfloat16, f'{i} f_ij dtype error inside'
                    # assert noise_emb.dtype == jnp.bfloat16, f'{i} noise_emb dtype error inside'
                    # assert s_mask.dtype == jnp.bool_, f'{i} s_mask dtype error inside'
                    
                    z_ij = self.edge_encoders[i](s_i, 
                                            self.edge_rbf_projection_out_cells[i](
                                                z_ij + self.edge_rbf_projection_in_cells[i](f_ij, noise_emb), noise_emb
                                            ), 
                                            s_mask, None, noise_emb)
                    z_ij = self.edge_layer_norms[i](z_ij)
        
        # assert z_ij.dtype == jnp.bfloat16, 'zij dtype error 2'
               
        if self.edge_decoding:
            # update positions & make scalar prediction
            z_ij = self.layer_norm_e(z_ij)
            phi_ij = self.edge_decoder(z_ij, noise_emb)
        
        ret = (s_i, z_ij)
        if self.predict_scalar_logit:
            ###### pooling on node features 
            if self.pooling == 'mean':
                mol_act = jnp.sum(s_i * s_mask.astype(self._dtype)[...,None], 
                                axis=0, keepdims=True) / \
                            (jnp.sum(s_mask.astype(self._dtype)) + 1e-6)
            else:
                mol_act = jnp.max(s_i - s_mask.astype(self._dtype)[..., None] * 1e5,
                                  axis=0, keepdims=True)
            mol_act = self.layer_norm_mol(mol_act)
            logits = self.mol_decoder(mol_act, noise_emb)[0]
            logit = logits[1] - logits[0]
            ret += (logit, )
              
        if self.predict_vector_pos:
            ##### convert phi_ij to safe dtype
            phi_ij = phi_ij.astype(self._safedtype)
            c_ij = jnp.expand_dims(c_ij, -1)
            dx_i = jnp.sum(c_ij * d_ij * phi_ij, 1)

            ret += (dx_i, )
            
        return ret
    
@_readout_register('gfn')
class AdaLNGFNReadout(nn.Module):

    config: Config

    def setup(self):
        
        self._dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        self._safedtype = jnp.float32 if SAFE_PRECISION_FLAG else self._dtype
        
        config = self.config
        self.unroll = config.settings.unroll
        self.n_interactions = config.settings.n_interactions
        self.stop_structure_grad = config.settings.stop_structure_grad
        self.stop_grad = jax.lax.stop_gradient if self.stop_structure_grad else lambda x:x
        
        self.share_weights = config.settings.share_weights
        
        self.noise_min = config.model.NoiseEmbedding.noise_min
        
        GFNIter = nn.checkpoint(AdaLNLoRAModulatedGFNIteration) if REMAT_FLAG \
                    else AdaLNLoRAModulatedGFNIteration
        
        self.gfn_iter_0 = GFNIter(config,
                                  predict_vector_pos=False,
                                  edge_decoding=False)
        if self.share_weights:
            self.gfn_iter = GFNIter(config)
        else:
            gfn_iters = []
            for k in range(self.n_interactions):
                gfn_iters.append(GFNIter(config))
            self.gfn_iters = gfn_iters
            
        self.with_contractive_bias = config.settings.with_contractive_bias
        if self.with_contractive_bias:
            self.contractive_force = jax.grad(self.contractive_energy)
            self.contractive_force_constant = config.settings.contractive_force_constant
        
    def contractive_energy(self, x, rg, s_mask):
        s_mask_  = s_mask[..., None].astype(self._safedtype)
        n_atoms = jnp.sum(s_mask_) + 1e-6
        x_center = jnp.sum(x * s_mask_, axis=-2, keepdims=True) / n_atoms
        x = x - x_center
        rg_x = jnp.sqrt(jnp.sum(jnp.sum((x * s_mask_) ** 2, axis=-1), axis=-1) / n_atoms)
        return self.contractive_force_constant * jnp.square(jax.nn.relu(rg_x - rg)) / rg * n_atoms
        
    def __call__(self, s_i, z_ij, x_i, s_mask, noise, rg, cond):
        x_i, noise, rg = jax.tree_map(lambda x:jnp.array(x, dtype=self._safedtype), 
                                     (x_i, noise, rg))
        
        s_i, z_ij = self.gfn_iter_0(s_i, z_ij, x_i, s_mask, noise, rg, cond)
        
        # assert s_i.dtype == jnp.bfloat16, "s_is dtype False"
        # assert z_ij.dtype == jnp.bfloat16, "z_ijs dtype False"
        
        displacements = []
        noise_base = noise
        for k in range(self.n_interactions):
            if self.share_weights:
                _, _, dx_i = \
                    self.gfn_iter(s_i, z_ij, self.stop_grad(x_i), 
                                s_mask, noise, rg, cond)
            else:
                _, _, dx_i = \
                    self.gfn_iters[k](s_i, z_ij, self.stop_grad(x_i), 
                                s_mask, noise, rg, cond)
            if self.with_contractive_bias:
                dx_i = dx_i + self.contractive_force(
                    jax.lax.stop_gradient(x_i), rg, s_mask)
            x_i = x_i - noise * dx_i
            
            # assert x_i.dtype == jnp.float32, "x_i dtype False"
            
            if not self.unroll:
                #### dx_i: normalized displacements #####
                #### probable bugs here: 
                #### 1. dx0 = gfn(x0)
                #### 2. x1 = x0 - noise * dx0
                #### 3. noise = noise * scaling_factor
                #### 4. dx1 = gfn(x1)
                #### 5. x2 = x1 - noise * dx1 
                #### So, dx should be: dx = dx0 + dx1 * scaling_factor
                displacements.append(dx_i + displacements[-1] \
                                                    if len(displacements) else dx_i)
            else:
                ### try this: 0605 (do not work well, why?)
                displacements.append(noise / noise_base * dx_i + displacements[-1] \
                                                     if len(displacements) else dx_i)
            
            noise = jnp.maximum(noise * 0.5, self.noise_min)
        
        return displacements
    
# @_readout_register('gfn')
# class GFNScalarReadout(nn.Module):

#     config: Config
    
#     def setup(self):
        
#         config = self.config
#         self.n_interactions = config.settings.n_interactions
#         gfn_layers = []
#         for i in range(self.n_interactions - 1):
#             gfn_iter = LoRAModulatedGFNIteration(config, 
#                                         predict_scalar_logit=False,
#                                         predict_vector_pos=False,
#                                         edge_encoding=True,
#                                         edge_decoding=False)
#             gfn_layers.append(gfn_iter)
            
#         gfn_layers.append(
#             LoRAModulatedGFNIteration(config, 
#                              predict_scalar_logit=True,
#                              predict_vector_pos=False,
#                              edge_encoding=False,
#                              edge_decoding=False)
#         )
#         self.gfn_layers = gfn_layers
        
#     def __call__(self, s_i, z_ij, x_i, s_mask, noise=None, Rg=None):
#         for k in range(self.n_interactions-1):
#             s_i, z_ij = self.gfn_layers[k](s_i, z_ij, x_i, s_mask, noise, Rg)
#         s_i, z_ij, logit = self.gfn_layers[-1](s_i, z_ij, x_i, s_mask, noise, Rg)
        
#         return logit    