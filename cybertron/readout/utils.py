# Basic utils for GFN modules.

import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Optional, Union, Callable
from ..common.layers.dense import Dense
from ..common.layers.mlp import MLP
from ..common.activation import ShiftedSoftplus

class MLPEdgeEncoder(nn.Module):

    dim_s: int
    dim_z: int
    dim_rbf: int
    dim_edge: int
    layer_dims: Optional[tuple] = None
    activation: Union[str, Callable] = 'relu'

    def setup(self):

        self.cin = self.dim_s + self.dim_z + self.dim_rbf
        self.cout = self.dim_edge

        self.mlp_encoder = MLP(output_sizes=self.layer_dims + (self.cout,), activation=self.activation)
        self.expand_dims = jnp.expand_dims
        
        # related to s
        self.dense_si = nn.Dense(features=self.dim_s)
        self.dense_s_ij = nn.Dense(features=self.dim_s)
        
        self.layer_norm = nn.LayerNorm()
        self.ssp = ShiftedSoftplus()
        self.dense_f_ij = nn.Sequential([Dense(dim_out=self.dim_rbf, activation=self.ssp), 
                                         Dense(dim_out=self.dim_rbf, activation=self.ssp)])

    def __call__(self, si, z_ij, f_ij):
        si = self.dense_si(si)
        # si_neighbours = gather_vectors(si, neighbors) # (B, A, A, Cs)
        s_ij = 0.5 * (self.expand_dims(si, -2) + self.expand_dims(si, -3)) # (B, A, A, Cs)
        s_ij = self.dense_s_ij(s_ij) # (B, A, A, Cs)
        
        f_ij = self.layer_norm(self.dense_f_ij(f_ij)) # (B, A, A, Cf)
        
        # s_ij: (B, A, N, Cs), z_ij: (B, A, A, Cz), f_ij: (B, A, A, Cf)
        input_feat = jnp.concatenate((s_ij, z_ij, f_ij), axis=-1)
        
        return self.mlp_encoder(input_feat)

class MLPEdgeDecoder(nn.Module):

    cin: int
    cout: int
    layer_dims: Optional[tuple] = None
    activation: Union[str, Callable] = 'relu'

    def setup(self):
        
        self.mlp_decoder = MLP(output_sizes=self.layer_dims + (self.cout,), activation=self.activation)

    def __call__(self, eij):
        return self.mlp_decoder(eij)