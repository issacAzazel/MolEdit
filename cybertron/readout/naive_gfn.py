
import jax
import math
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from typing import Union, Callable

from jax import lax
from flax.linen.initializers import lecun_normal, zeros_init
from ..common.config_load import Config
from ..common.layers.dense import Dense, LoRAModulatedDense
from ..common.layers.mlp import MLP
from ..common.cutoff import get_cutoff, GaussianCutoff
from ..common.rbf import GaussianBasis
from ..common.activation import get_activation
from ..model.interaction.schnet_interaction import SchnetInteraction

class NaiveMLPEdgeEncoder(nn.Module):
    num_rbf_basis: int 
    dim_atom_feature: int 
    dim_edge_feature: int = 64 
    activation: Union[str, Callable] = "relu"
    
    def setup(self):
        self.mlp_encoder_h = MLP(
            output_sizes=[self.dim_edge_feature, self.dim_edge_feature], 
            activation=self.activation)
        self.mlp_encoder_x = MLP(
            output_sizes=[self.dim_edge_feature, self.dim_edge_feature],
            activation=self.activation
        )
        
    def __call__(self, hij, xij, mask_2d):
        return self.mlp_encoder_h(hij) * self.mlp_encoder_x(xij) * \
            jnp.expand_dims(mask_2d.astype(jnp.float32), -1)
    
class NaiveMLPConditionalEdgeEncoder(nn.Module):
    num_rbf_basis: int 
    dim_atom_feature: int 
    dim_bond_feature: int = 64
    dim_edge_feature: int = 64 
    num_bond_types: int = 4 
    activation: Union[str, Callable] = "relu"
    
    def setup(self):
        self.mlp_encoder_h = MLP(
            output_sizes=[self.dim_edge_feature, self.dim_edge_feature], 
            activation=self.activation)
        self.mlp_encoder_x = MLP(
            output_sizes=[self.dim_edge_feature, self.dim_edge_feature],
            activation=self.activation
        )
        self.bond_embedding = nn.Embed(
            num_embeddings=self.num_bond_types,
            features=self.dim_bond_feature,
            dtype=jnp.float32
        )
        self.mlp_encoder_c = MLP(
            output_sizes=[self.dim_edge_feature, self.dim_edge_feature],
            activation=self.activation
        )
        
    def __call__(self, hij, xij, bij, mask_2d):
        cij = self.bond_embedding(bij)
        return self.mlp_encoder_h(hij) * self.mlp_encoder_x(xij) * \
            self.mlp_encoder_c(cij) * jnp.expand_dims(mask_2d.astype(jnp.float32), -1)
            
class NaiveMLPEdgeDecoder(nn.Module):
    dim_edge_feature: int = 64
    activation: Union[str, Callable] = "relu"
    
    def setup(self):
        self.mlp_decoder = MLP(
            output_sizes=[self.dim_edge_feature, 1],
            activation=self.activation)
        
    def __call__(self, mij, mask_2d):
        return self.mlp_decoder(mij) * jnp.expand_dims(mask_2d.astype(jnp.float32), -1)

class NaiveGraphFieldNetwork(nn.Module):
    num_atoms: int = 9
    num_atom_types: int = 64 
    dim_atom_feature: int = 64 
    dim_atom_filter: int = 64 
    num_rbf_basis: int = 64 
    dim_edge_feature: int = 64
    edge_activation: Union[str, Callable] = "relu"
    atom_activation: Union[str, Callable] = "ssp"
    n_interactions: int = 3
    cutoff: float = 10.0
    epsilon: float = 1e-6
    
    def setup(self):
        self.rbf = GaussianBasis(
            r_max=self.cutoff, 
            sigma=0.3, 
            delta=None,
            num_basis=self.num_rbf_basis,
        )
        
        self.edge_encoder_cells = [
            NaiveMLPEdgeEncoder(num_rbf_basis=self.num_rbf_basis,
                                dim_atom_feature=self.dim_atom_feature,
                                dim_edge_feature=self.dim_edge_feature,
                                activation=self.edge_activation)
            for i in range(self.n_interactions)
        ]
        
        self.atom_encoder_cells = [
            SchnetInteraction(dim_filter=self.dim_atom_filter,
                              activation=self.atom_activation)
            for i in range(self.n_interactions)
        ]
        
        self.edge_decoder_cells = [
            NaiveMLPEdgeDecoder(dim_edge_feature=self.dim_edge_feature,
                                activation=self.edge_activation) 
            for i in range(self.n_interactions)
        ]
        
        self.atom_embedding = nn.Embed(
            num_embeddings=self.num_atom_types,
            features=self.dim_atom_feature, 
            dtype=jnp.float32
        )
        
    def __call__(self, positions, atom_types):
        xi = positions 
        hi = self.atom_embedding(atom_types)
        
        atom_mask = atom_types > 0 
        edge_mask = jnp.logical_and(jnp.expand_dims(atom_mask, -1),
                                    jnp.expand_dims(atom_mask, -2))
        self_mask = jnp.logical_not(jnp.eye(self.num_atoms, dtype=jnp.bool_))
        edge_mask = jnp.logical_and(edge_mask, self_mask)
        
        for i in range(self.n_interactions):
            d_ij = jnp.expand_dims(xi, 0) - jnp.expand_dims(xi, 1)
            r_ij = jnp.linalg.norm(
                d_ij + self.epsilon * \
                    (1.0 - jnp.expand_dims(edge_mask, -1).astype(jnp.float32)),
                axis=-1)
            
            hi = hi / jnp.linalg.norm(
                hi + self.epsilon * \
                    (1.0 - jnp.expand_dims(atom_mask, -1).astype(jnp.float32)),
                axis=-1, keepdims=True
            ) + self.epsilon
            h_ij = 0.5 * (jnp.expand_dims(hi, 0) + jnp.expand_dims(hi, 1))
            
            f_ij = self.rbf(r_ij)
            m_ij = self.edge_encoder_cells[i](h_ij, f_ij, edge_mask)
            
            hi, _ = self.atom_encoder_cells[i](
                hi, atom_mask, m_ij, edge_mask, jnp.ones_like(r_ij)
            )
            
            phi_ij = self.edge_decoder_cells[i](m_ij, edge_mask)
            
            numerator = jnp.expand_dims(r_ij, -1) + \
                self.epsilon * (1.0 - jnp.expand_dims(edge_mask, -1).astype(jnp.float32))
    
            dxi = jnp.sum(jnp.expand_dims(edge_mask, -1).astype(jnp.float32) / \
                numerator * d_ij * phi_ij, axis=1)
            
            xi = xi + dxi 
    
        return xi - positions

class NaiveGraphFieldConditionalNetwork(nn.Module):
    num_atoms: int = 9
    num_atom_types: int = 64 
    num_bond_types: int = 4
    dim_atom_feature: int = 64
    dim_atom_filter: int = 64 
    dim_bond_feature: int = 64
    num_rbf_basis: int = 64 
    dim_edge_feature: int = 64
    edge_activation: Union[str, Callable] = "relu"
    atom_activation: Union[str, Callable] = "ssp"
    n_interactions: int = 3
    cutoff: float = 10.0
    epsilon: float = 1e-6
    
    def setup(self):
        self.rbf = GaussianBasis(
            r_max=self.cutoff, 
            sigma=0.3, 
            delta=None,
            num_basis=self.num_rbf_basis,
        )
        
        self.edge_encoder_cells = [
            NaiveMLPConditionalEdgeEncoder(num_rbf_basis=self.num_rbf_basis,
                                           dim_atom_feature=self.dim_atom_feature,
                                           dim_bond_feature=self.dim_bond_feature,
                                           dim_edge_feature=self.dim_edge_feature,
                                           num_bond_types=self.num_bond_types,
                                           activation=self.edge_activation)
            for i in range(self.n_interactions)
        ]
        
        self.atom_encoder_cells = [
            SchnetInteraction(dim_filter=self.dim_atom_filter,
                              activation=self.atom_activation)
            for i in range(self.n_interactions)
        ]
        
        self.edge_decoder_cells = [
            NaiveMLPEdgeDecoder(dim_edge_feature=self.dim_edge_feature,
                                activation=self.edge_activation) 
            for i in range(self.n_interactions)
        ]
        
        self.atom_embedding = nn.Embed(
            num_embeddings=self.num_atom_types,
            features=self.dim_atom_feature, 
            dtype=jnp.float32
        )
        
    def __call__(self, positions, atom_types, bond_types):
        xi = positions 
        hi = self.atom_embedding(atom_types)
        
        atom_mask = atom_types > 0 
        edge_mask = jnp.logical_and(jnp.expand_dims(atom_mask, -1),
                                    jnp.expand_dims(atom_mask, -2))
        self_mask = jnp.logical_not(jnp.eye(self.num_atoms, dtype=jnp.bool_))
        edge_mask = jnp.logical_and(edge_mask, self_mask)
        
        for i in range(self.n_interactions):
            d_ij = jnp.expand_dims(xi, 0) - jnp.expand_dims(xi, 1)
            r_ij = jnp.linalg.norm(
                d_ij + self.epsilon * \
                    (1.0 - jnp.expand_dims(edge_mask, -1).astype(jnp.float32)),
                axis=-1)
            
            hi = hi / jnp.linalg.norm(
                hi + self.epsilon * \
                    (1.0 - jnp.expand_dims(atom_mask, -1).astype(jnp.float32)),
                axis=-1, keepdims=True
            ) + self.epsilon
            h_ij = 0.5 * (jnp.expand_dims(hi, 0) + jnp.expand_dims(hi, 1))
            
            f_ij = self.rbf(r_ij)
            m_ij = self.edge_encoder_cells[i](h_ij, f_ij, bond_types, edge_mask)
            
            hi, _ = self.atom_encoder_cells[i](
                hi, atom_mask, m_ij, edge_mask, jnp.ones_like(r_ij)
            )
            
            phi_ij = self.edge_decoder_cells[i](m_ij, edge_mask)
            
            numerator = jnp.expand_dims(r_ij, -1) + \
                self.epsilon * (1.0 - jnp.expand_dims(edge_mask, -1).astype(jnp.float32))
    
            dxi = jnp.sum(jnp.expand_dims(edge_mask, -1).astype(jnp.float32) / \
                numerator * d_ij * phi_ij, axis=1)
            
            xi = xi + dxi 
    
        return xi - positions