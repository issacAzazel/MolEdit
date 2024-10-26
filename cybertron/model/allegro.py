"""Allegro Model on JAX"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import e3nn_jax as e3nn

from typing import Callable, List, Optional, Union
from e3nn_jax import Irreps

from ..modules.allegro_mlp import MultiLayerPerceptron, LoRAModulatedMultiLayerPerceptron
from ..modules.allegro_linear import Linear 
from ..modules.allegro_batch_norm import BatchNorm
from ..common.activation import get_activation
from cybertron.modules.basic import ActFuncWrapper

from config.global_setup import EnvironConfig
global_setup = EnvironConfig() ### Set Hyper-parameters here
NORM_SMALL = global_setup.norm_small
BF16_FLAG = global_setup.bf16_flag
DROPOUT_FLAG = global_setup.use_dropout

def filter_layers(layer_irreps: List[e3nn.Irreps], max_ell: int) -> List[e3nn.Irreps]:
    layer_irreps = list(layer_irreps)
    filtered = [e3nn.Irreps(layer_irreps[-1])]
    for irreps in reversed(layer_irreps[:-1]):
        irreps = e3nn.Irreps(irreps)
        irreps = irreps.filter(
            keep=e3nn.tensor_product(
                filtered[0],
                e3nn.Irreps.spherical_harmonics(lmax=max_ell),
            ).regroup()
        )
        filtered.insert(0, irreps)
    return filtered

class AllegroLayer(nn.Module):

    avg_num_neighbors: float
    max_ell: int
    env_irreps: e3nn.Irreps
    output_irreps: e3nn.Irreps
    shared_weights_flag: bool = False
    mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu
    mlp_n_hidden: int = 64
    mlp_n_layers: int = 3
    p: int = 6
    gradient_normalization: str = "path"

    @nn.compact
    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        x: jnp.ndarray,             # [n_edges, features]     ## edge features
        V: e3nn.IrrepsArray,        # [n_edges, irreps]       ## features 
        u: jnp.ndarray,             # [n_edges] edge cutoff   ## edge cutoff
        m: jnp.ndarray,             # [n_edges] edge mask     ## edge mask
        senders: jnp.ndarray,       # [n_edges]               ## edge senders
    ) -> e3nn.IrrepsArray:
        
        _dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        
        num_edges = vectors.shape[0]
        assert vectors.shape == (num_edges, 3)
        assert x.shape == (num_edges, x.shape[-1])
        assert V.shape == (num_edges, V.irreps.dim)
        assert senders.shape == (num_edges,)

        x = x * m[:, None]
        irreps_out = e3nn.Irreps(self.output_irreps)
        irreps_env = e3nn.Irreps(self.env_irreps)
        n_neighbors = e3nn.scatter_sum(m, dst=senders, map_back=True)

        ## difference between two methods: w method 01 is same between different l's, and in method 02 is different. 
        if self.shared_weights_flag: ## method 01
            w = MultiLayerPerceptron((V.irreps.mul_gcd,), 
                                     gradient_normalization=self.gradient_normalization)(x) ## greatest common divisor of the multiplicities
            Y = e3nn.spherical_harmonics(range(self.max_ell + 1), vectors, True)

            # wY = e3nn.scatter_sum(
            #     w[:, :, None] * Y[:, None, :] * m[:, None, None], dst=senders, map_back=True ## add edge mask
            # ) / jnp.sqrt(self.avg_num_neighbors) ## original method: different from mindspore

            wY = e3nn.scatter_sum(
                w[:, :, None] * Y[:, None, :] * m[:, None, None], dst=senders, map_back=True
            ) / (n_neighbors[:, None, None] + 1e-5)

        else: ## method 02
            w = MultiLayerPerceptron((irreps_env.num_irreps,), 
                                     gradient_normalization=self.gradient_normalization)(x) ## sum of irreps
            Y = e3nn.spherical_harmonics(irreps_env, vectors, True)
            wY = e3nn.scatter_sum(
                w * Y * m[:, None], dst=senders, map_back=True
            ) / (n_neighbors[:, None] + 1e-5)
            wY = wY.mul_to_axis()       

        assert wY.shape == (num_edges, V.irreps.mul_gcd, wY.irreps.dim)

        V = e3nn.tensor_product(
            wY, V.mul_to_axis(), filter_ir_out="0e" + irreps_out ## (N_edge, N_channel, N_irreps)
        ).axis_to_mul()

        if "0e" in V.irreps:
            x = jnp.concatenate([x, V.filter(keep="0e").array], axis=1)
            V = V.filter(drop="0e") ## different from mindspore

        x = MultiLayerPerceptron(
            (self.mlp_n_hidden,) * self.mlp_n_layers,
            self.mlp_activation,
            output_activation=False,
            gradient_normalization=self.gradient_normalization,
        )(x)
        lengths = e3nn.norm(vectors).array
        x = u[:, None] * x
        assert x.shape == (num_edges, self.mlp_n_hidden)

        V = Linear(irreps_out)(V)
        assert V.shape == (num_edges, V.irreps.dim)

        return (x, V)
    
class LoRAModulatedAllegroLayer(nn.Module):

    avg_num_neighbors: float
    max_ell: int
    env_irreps: e3nn.Irreps
    output_irreps: e3nn.Irreps
    shared_weights_flag: bool = False
    mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu
    mlp_n_hidden: int = 64
    mlp_n_layers: int = 3
    p: int = 6
    gradient_normalization: str = "path"
    dropout_rate: float = 0.0
    lora_rank: int = 4
    lora_alpha: Union[int, None] = None
    lora_dropout_rate: float = 0.0 

    @nn.compact
    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        x: jnp.ndarray,             # [n_edges, features]     ## edge features
        V: e3nn.IrrepsArray,        # [n_edges, irreps]       ## features 
        u: jnp.ndarray,             # [n_edges] edge cutoff   ## edge cutoff
        m: jnp.ndarray,             # [n_edges] edge mask     ## edge mask
        senders: jnp.ndarray,       # [n_edges]               ## edge senders
        modulated_params: jnp.ndarray
    ) -> e3nn.IrrepsArray:
        
        num_edges = vectors.shape[0]
        assert vectors.shape == (num_edges, 3)
        assert x.shape == (num_edges, x.shape[-1])
        assert V.shape == (num_edges, V.irreps.dim)
        assert senders.shape == (num_edges,)

        x = x * m[:, None]
        irreps_out = e3nn.Irreps(self.output_irreps)
        irreps_env = e3nn.Irreps(self.env_irreps)
        n_neighbors = e3nn.scatter_sum(m, dst=senders, map_back=True)

        ## difference between two methods: w method 01 is same between different l's, and in method 02 is different. 
        if self.shared_weights_flag: ## method 01
            w = LoRAModulatedMultiLayerPerceptron(
                    list_neurons = (V.irreps.mul_gcd,), 
                    act = self.mlp_activation,
                    gradient_normalization = self.gradient_normalization,
                    output_activation = False,
                    dropout_rate = self.dropout_rate,
                    lora_rank = self.lora_rank, 
                    lora_alpha = self.lora_alpha,
                    lora_dropout_rate = self.lora_dropout_rate)(x, modulated_params) ## greatest common divisor of the multiplicities
            Y = e3nn.spherical_harmonics(range(self.max_ell + 1), vectors, True)

            # wY = e3nn.scatter_sum(
            #     w[:, :, None] * Y[:, None, :] * m[:, None, None], dst=senders, map_back=True ## add edge mask
            # ) / jnp.sqrt(self.avg_num_neighbors) ## original method: different from mindspore

            wY = e3nn.scatter_sum(
                w[:, :, None] * Y[:, None, :] * m[:, None, None], dst=senders, map_back=True
            ) / (n_neighbors[:, None, None] + 1e-5)

        else: ## method 02
            w = LoRAModulatedMultiLayerPerceptron(
                    list_neurons = (irreps_env.num_irreps,), 
                    act = self.mlp_activation,
                    gradient_normalization = self.gradient_normalization,
                    output_activation = False,
                    dropout_rate = self.dropout_rate,
                    lora_rank = self.lora_rank, 
                    lora_alpha = self.lora_alpha,
                    lora_dropout_rate = self.lora_dropout_rate)(x, modulated_params) ## sum of irreps
            Y = e3nn.spherical_harmonics(irreps_env, vectors, True)
            wY = e3nn.scatter_sum(
                w * Y * m[:, None], dst=senders, map_back=True
            ) / (n_neighbors[:, None] + 1e-5)
            wY = wY.mul_to_axis()       

        assert wY.shape == (num_edges, V.irreps.mul_gcd, wY.irreps.dim)

        V = e3nn.tensor_product(
            wY, V.mul_to_axis(), filter_ir_out="0e" + irreps_out ## (N_edge, N_channel, N_irreps)
        ).axis_to_mul()

        if "0e" in V.irreps:
            x = jnp.concatenate([x, V.filter(keep="0e").array], axis=1)
            V = V.filter(drop="0e") ## different from mindspore
        
        x = LoRAModulatedMultiLayerPerceptron(
                list_neurons = (self.mlp_n_hidden,) * self.mlp_n_layers,
                act = self.mlp_activation,
                gradient_normalization = self.gradient_normalization,
                output_activation = False,
                dropout_rate = self.dropout_rate,
                lora_rank = self.lora_rank, 
                lora_alpha = self.lora_alpha,
                lora_dropout_rate = self.lora_dropout_rate)(x, modulated_params)
        # x = MultiLayerPerceptron(
        #     (self.mlp_n_hidden,) * self.mlp_n_layers,
        #     self.mlp_activation,
        #     output_activation=False,
        #     gradient_normalization=self.gradient_normalization,
        # )(x)
        lengths = e3nn.norm(vectors).array
        x = u[:, None] * x
        assert x.shape == (num_edges, self.mlp_n_hidden)

        V = Linear(irreps_out)(V)
        assert V.shape == (num_edges, V.irreps.dim)

        return (x, V)

class Allegro(nn.Module):
    r"""Allegro Layer"""

    avg_num_neighbors: int
    max_ell: int
    irreps: Irreps ## env embedding irreps
    mlp_activation: Union[str, Callable] = nn.silu
    mlp_n_hidden: int = 256
    mlp_n_layers: int = 2
    env_n_channel: int = 32
    output_n_channel: int = 128
    output_irreps: Irreps = Irreps("0e")
    num_layers: int = 3
    eps: float = 1e-3
    shared_weights_flag: bool = False
    gradient_normalization: str = "path"

    @nn.compact
    def __call__(self, node_vec, edge_vec, pos_vec, pos_length, edge_cutoff, edge_mask, edge_index, node_mask, add_edge_feats=None):
        r"""
        Input: 
        -------------------------------------
            node_vec: (A, Fn)
            edge_vec: (A*(A-1), Fe)
            pos_vec: (A*(A-1), 3)
            edge_cutoff: (A*(A-1),)
            edge_mask: (A*(A-1),)
        """

        num_edges = edge_vec.shape[0]
        num_nodes = node_vec.shape[0]
        assert pos_vec.shape == (num_edges, 3)
        assert edge_index.shape == (2, num_edges)

        irreps = self.env_n_channel * Irreps(self.irreps)
        irreps_out = self.output_n_channel * Irreps(self.output_irreps)
        irreps_layers = [irreps] * self.num_layers

        irreps_layers = [irreps] * self.num_layers + [irreps_out]
        irreps_layers = filter_layers(irreps_layers, self.max_ell)

        # vectors = vectors / self.radial_cutoff
        senders = edge_index[0]
        receivers = edge_index[1]

        edge_vec = edge_vec * edge_mask[:, None]
        node_vec = node_vec * node_mask[:, None]
        pos_vec = e3nn.IrrepsArray("1o", pos_vec + self.eps)

        d = pos_length # (N_edge,)
        x = jnp.concatenate(
            [
                edge_vec,                # (N_edge, Fe)
                node_vec[senders],       # (N_edge, Fn)
                node_vec[receivers],     # (N_edge, Fn)
            ],
            axis=-1,
        )

        # Protection against exploding dummy edges
        # set to zero if d == 0
        x = jnp.where(d[:, None] == 0.0, 0.0, x)

        x = MultiLayerPerceptron(
            (
                self.mlp_n_hidden // 8,
                self.mlp_n_hidden // 4,
                self.mlp_n_hidden // 2,
                self.mlp_n_hidden,
            ),
            self.mlp_activation,
            output_activation=False,
            gradient_normalization=self.gradient_normalization,
        )(x)
        x = edge_cutoff[:, None] * x # (N_edge, F)
        assert x.shape == (num_edges, self.mlp_n_hidden)

        irreps_Y = irreps_layers[0].filter( ## MulIrrep: (mul, ir(l, p))
            keep=lambda mir: pos_vec.irreps[0].ir.p ** mir.ir.l == mir.ir.p ## (-1) ** l = p
        )
        V = e3nn.spherical_harmonics(irreps_Y, pos_vec, True)

        if add_edge_feats is not None:
            V = e3nn.concatenate([V, add_edge_feats])

        num_env_irreps = V.irreps.num_irreps
        w = MultiLayerPerceptron((num_env_irreps,), act=self.mlp_activation, gradient_normalization=self.gradient_normalization)(x)
        V = w * V ## auto broadcast to irreps
        assert V.shape == (num_edges, V.irreps.dim)

        for irreps in irreps_layers[1:]:
            
            ## optional: add norm
            # x = nn.LayerNorm()(x)
            # V = e3nn.flax.BatchNorm(instance=True)(V)

            y, V = AllegroLayer(
                avg_num_neighbors=self.avg_num_neighbors,
                max_ell=self.max_ell,
                env_irreps=irreps_Y,
                output_irreps=irreps,
                mlp_activation=self.mlp_activation,
                mlp_n_hidden=self.mlp_n_hidden,
                mlp_n_layers=self.mlp_n_layers,
                shared_weights_flag=self.shared_weights_flag,
                gradient_normalization=self.gradient_normalization
            )(pos_vec, x, V, edge_cutoff, edge_mask, senders)

            alpha = 0.5
            x = (x + alpha * y) / jnp.sqrt(1 + alpha**2)

        x = MultiLayerPerceptron((128,), 
                                 gradient_normalization=self.gradient_normalization)(x) ## Liyh: need to add args

        xV = Linear(irreps_out, 
                              gradient_normalization=self.gradient_normalization)(e3nn.concatenate([x, V]))

        if xV.irreps != irreps_out:
            raise ValueError(f"output_irreps {irreps_out} is not reachable")

        return xV
    

class LoRAModulatedAllegro(nn.Module):
    r"""Allegro Layer"""

    avg_num_neighbors: int
    max_ell: int
    irreps: Irreps ## env embedding irreps
    mlp_activation: Union[str, Callable] = nn.silu
    mlp_n_hidden: int = 256
    mlp_n_layers: int = 2
    env_n_channel: int = 32
    output_n_channel: int = 128
    output_irreps: Irreps = Irreps("0e")
    num_layers: int = 3
    layer_norm_reps: bool = False
    eps: float = 1e-3
    shared_weights_flag: bool = False
    gradient_normalization: str = "path"
    dropout_rate: float = 0.0
    lora_rank: int = 4
    lora_alpha: Union[int, None] = None
    lora_dropout_rate: float = 0.0 

    @nn.compact
    def __call__(self, node_vec, edge_vec, pos_vec, pos_length, edge_cutoff, edge_mask, edge_index, node_mask, modulated_params, add_edge_feats=None):
        r"""
        Input: 
        -------------------------------------
            node_vec: (A, Fn)
            edge_vec: (A*(A-1), Fe)
            pos_vec: (A*(A-1), 3)
            edge_cutoff: (A*(A-1),)
            edge_mask: (A*(A-1),)
        """
        
        _dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32

        num_edges = edge_vec.shape[0]
        num_nodes = node_vec.shape[0]
        assert pos_vec.shape == (num_edges, 3)
        assert edge_index.shape == (2, num_edges)

        irreps = self.env_n_channel * Irreps(self.irreps)
        irreps_out = self.output_n_channel * Irreps(self.output_irreps)
        irreps_layers = [irreps] * self.num_layers

        irreps_layers = [irreps] * self.num_layers + [irreps_out]
        irreps_layers = filter_layers(irreps_layers, self.max_ell)

        # vectors = vectors / self.radial_cutoff
        senders = edge_index[0]
        receivers = edge_index[1]

        edge_vec = edge_vec * edge_mask[:, None]
        node_vec = node_vec * node_mask[:, None]
        pos_vec = e3nn.IrrepsArray("1o", pos_vec + self.eps)

        d = pos_length # (N_edge,)
        x = jnp.concatenate(
            [
                edge_vec,                # (N_edge, Fe)
                node_vec[senders],       # (N_edge, Fn)
                node_vec[receivers],     # (N_edge, Fn)
            ],
            axis=-1,
        )

        # Protection against exploding dummy edges
        # set to zero if d == 0
        x = jnp.where(d[:, None] == 0.0, 0.0, x)

        x = MultiLayerPerceptron(
            (
                self.mlp_n_hidden // 8,
                self.mlp_n_hidden // 4,
                self.mlp_n_hidden // 2,
                self.mlp_n_hidden,
            ),
            self.mlp_activation,
            output_activation=False,
            gradient_normalization=self.gradient_normalization,
        )(x)
        x = edge_cutoff[:, None] * x # (N_edge, F)
        assert x.shape == (num_edges, self.mlp_n_hidden)

        irreps_Y = irreps_layers[0].filter( ## MulIrrep: (mul, ir(l, p))
            keep=lambda mir: pos_vec.irreps[0].ir.p ** mir.ir.l == mir.ir.p ## (-1) ** l = p
        )
        V = e3nn.spherical_harmonics(irreps_Y, pos_vec, True)

        if add_edge_feats is not None:
            V = e3nn.concatenate([V, add_edge_feats])

        num_env_irreps = V.irreps.num_irreps
        w = MultiLayerPerceptron((num_env_irreps,), act=self.mlp_activation, gradient_normalization=self.gradient_normalization)(x)
        V = w * V ## auto broadcast to irreps
        assert V.shape == (num_edges, V.irreps.dim)

        for irreps in irreps_layers[1:]:
            
            ## optional: add norm
            if self.layer_norm_reps:
                x = ActFuncWrapper(nn.LayerNorm(epsilon=NORM_SMALL, dtype=_dtype, param_dtype=jnp.float32))(x)
                V = BatchNorm(instance=True)(V)

            y, V = LoRAModulatedAllegroLayer(
                avg_num_neighbors=self.avg_num_neighbors,
                max_ell=self.max_ell,
                env_irreps=irreps_Y,
                output_irreps=irreps,
                mlp_activation=self.mlp_activation,
                mlp_n_hidden=self.mlp_n_hidden,
                mlp_n_layers=self.mlp_n_layers,
                shared_weights_flag=self.shared_weights_flag,
                gradient_normalization=self.gradient_normalization, 
                dropout_rate = self.dropout_rate,
                lora_rank = self.lora_rank, 
                lora_alpha = self.lora_alpha,
                lora_dropout_rate = self.lora_dropout_rate
            )(pos_vec, x, V, edge_cutoff, edge_mask, senders, modulated_params)

            alpha = 0.5
            x = (x + alpha * y) / jnp.sqrt(1 + alpha**2)

        x = MultiLayerPerceptron((128,), 
                                 gradient_normalization=self.gradient_normalization)(x) ## Liyh: need to add args

        xV = Linear(irreps_out, 
                              gradient_normalization=self.gradient_normalization)(e3nn.concatenate([x, V]))

        if xV.irreps != irreps_out:
            raise ValueError(f"output_irreps {irreps_out} is not reachable")

        return xV