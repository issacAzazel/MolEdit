# Basic utils for GFN modules.

import jax
import jax.numpy as jnp
import flax.linen as nn
import math

from typing import Optional, Union, Callable
from ..common.layers.dense import Dense
from ..common.layers.mlp import MLP
from ..common.activation import ShiftedSoftplus
from cybertron.modules.basic import ActFuncWrapper

from config.global_setup import EnvironConfig
global_setup = EnvironConfig() ### Set Hyper-parameters here
NORM_SMALL = global_setup.norm_small
BF16_FLAG = global_setup.bf16_flag
SAFE_PRECISION_FLAG = global_setup.safe_precision_flag
DROPOUT_FLAG = global_setup.use_dropout

class MLPEdgeEncoder(nn.Module):

    dim_s: int
    dim_z: int
    dim_rbf: int
    dim_edge: int
    layer_dims: Optional[tuple] = None
    activation: Union[str, Callable] = 'relu'

    def setup(self):
        
        self._dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        
        self.cin = self.dim_s + self.dim_z + self.dim_rbf
        self.cout = self.dim_edge

        self.mlp_encoder = MLP(output_sizes=self.layer_dims + (self.cout,), activation=self.activation)
        self.expand_dims = jnp.expand_dims
        
        # related to s
        self.dense_si = nn.Dense(features=self.dim_s, 
                                 dtype=self._dtype, param_dtype=jnp.float32)
        self.dense_s_ij = nn.Dense(features=self.dim_s,
                                   dtype=self._dtype, param_dtype=jnp.float32)
        
        self.layer_norm = ActFuncWrapper(nn.LayerNorm(epsilon=NORM_SMALL, dtype=self._dtype, param_dtype=jnp.float32))
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
    
class LogGaussianBasisWithFlexRmax(nn.Module):
    
    r_min: float = 0.4
    sigma: float = 0.3
    num_basis: Optional[int] = 64
    rescale: bool = True
    clip_distance: bool = False
    r_ref: float = 10.0
    eps: float = 1e-6
    dtype: str = jnp.float32

    def setup(self):

        # self._dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        # self._dtype = jnp.float32 if SAFE_PRECISION_FLAG else self._dtype
        
        self.coefficient = -0.5 * 1.0 / (math.pow(self.sigma, 2))
        self.inv_ref = 1.0 / (self.r_ref)
        self.log_rmin = math.log(self.r_min  * self.inv_ref)

    def __call__(self, distance: jnp.ndarray, r_max: jnp.ndarray) -> jnp.ndarray:
        # distance (...)
        # rmax: float, shape=()
        if self.clip_distance:
            distance = jnp.clip(distance, self.r_min, r_max)
        
        # (...,) -> (..., 1)
        log_r = jnp.log(distance * self.inv_ref + self.eps)
        log_r = jnp.expand_dims(log_r, axis=-1)
        
        offsets = jnp.linspace(
            self.log_rmin, jnp.log(r_max * self.inv_ref), 
            self.num_basis, dtype=self.dtype)
        offsets = jnp.reshape(offsets, (1,) * len(distance.shape) + (self.num_basis, ))
        
        # (..., 1) - (..., K) -> (..., K)
        log_diff = log_r - offsets
        rbf = jnp.exp(self.coefficient * jnp.square(log_diff))

        if self.rescale:
            rbf = rbf * 2.0 - 1.0
        
        return rbf
    
class BesselBasisWithFlexRmax(nn.Module):
    r"""Bessel type RBF.
    ## Args:
        r_max (Length):         Maximum distance.

        num_basis (int):        Number of basis functions. Defatul: None.

        name (str):             Name of the module. Default: "bessel_basis".

    """
    num_basis: int = 8
    trainable: bool = True
    tol: float = 1e-3
    name: str = "bessel_basis"
    dtype: str = jnp.float32

    @nn.compact
    def __call__(self, distance: jax.Array, r_max: jax.Array) -> jax.Array:
        
        # _dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        # _dtype = jnp.float32 if SAFE_PRECISION_FLAG else _dtype
        
        d_min = math.sqrt(self.tol * 6.0)

        prefactor = jnp.sqrt(2.0 / r_max)
        # prefactor = r_max / 2.0

        bessel_weights = jnp.linspace(1.0, self.num_basis, self.num_basis,
                                      dtype=jnp.float32) * jnp.pi
        if self.trainable:
            bessel_weights = self.param("bessel_weights", nn.initializers.constant(bessel_weights), (self.num_basis,))
        else:
            bessel_weights = self.param("bessel_weights", nn.initializers.constant(bessel_weights), (self.num_basis,))
            bessel_weights = jax.lax.stop_gradient(bessel_weights)
        
        # (..., ) -> (..., 1)
        distance = jnp.expand_dims(distance, axis=-1)
        # (..., 1) -> (..., num_basis)
        bessel_distance = bessel_weights.astype(self.dtype) * distance
        bessel_distance = bessel_distance / r_max
        alpha = bessel_weights / r_max
        numerator = jnp.sin(bessel_distance)
        ret = jnp.where(distance < d_min, 
                        alpha*(1.0 - jnp.square(alpha*distance)/6.0),
                        numerator/jnp.maximum(distance, d_min))
        # ret = prefactor * (numerator / distance)
        ret = prefactor * ret

        return ret

class LogSinusoidalNoiseEmb(nn.Module):
    num_basis: int = 64
    clip_min_max: bool = False 
    eps: float = 1e-6
    dtype: str = jnp.float32
    
    def setup(self):
        # self._dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        # assert self.num_basis % 2 == 0, "num_basis should be even"
        # num_basis = self.num_basis // 2
        
        self.npi = jnp.linspace(0, self.num_basis-1, self.num_basis, dtype=self.dtype) * jnp.pi # (K, )
        self.pi = jnp.pi
        # self.npi_odd = 2 * self.npi + self.pi 
        # self.npi_even = 2 * (self.npi + self.pi)
        self.prefactor = (2 * self.npi + self.pi) * 0.5  
    
    def __call__(self, sigma: jnp.array, sigma_min: jnp.array, sigma_max: jnp.array):
        log_sigma = jnp.log(sigma)
        log_max = jnp.log(sigma_max)
        log_min = jnp.log(sigma_min + self.eps)
 
        offset = -log_min 
        scale = 1.0 / (log_max - log_min)
        log_sigma = (log_sigma + offset) * scale
        
        if self.clip_min_max:
            log_sigma = jnp.clip(log_sigma, 0.0, 1.0)
            
        return jnp.sin(self.prefactor * log_sigma)
    
class FourierBasis(nn.Module):
    num_basis: int = 64
    clamp_input: bool = True
    scale: str = 'normal' # 'log'
    eps: float = 1e-6
    decay: str = None ## 'left', 'right', 'double'
    dtype: str = jnp.float32
    
    def setup(self):
        # self._dtype = jnp.bfloat16 if BF16_FLAG else jnp.float32
        assert self.num_basis % 2 == 0, "num_basis should be even"
        num_basis = self.num_basis // 2
        
        self.npi = jnp.linspace(1, num_basis, num_basis, dtype=self.dtype) * jnp.pi # (K, )
        self.pi = jnp.pi
        
        assert self.decay in [None, 'left', 'right', 'double'], 'unsupported decay {}'.format(self.decay)
        assert self.scale in ['normal', 'log'], 'unsupported scale {}'.format(self.scale)
    
    def soft_clamp(self, x):
        # [-\infty, infty] -> [0, 1]
        # return x_max * nn.tanh(x/x_max)
        return 0.5 * (nn.tanh(2 * x - 1.0) + 1.0)
    
    def decay_fn(self, x): 
        #### x is normalized to [0, 1]
        
        if self.decay:
            if self.decay == 'right':
                decay = 0.5 * (1.0 + jnp.cos(jnp.pi * x))
            if self.decay == 'left':
                decay = 0.5 * (1.0 - jnp.cos(jnp.pi * x))
            if self.decay == 'double':
                decay = 0.5 * (1.0 + jnp.cos(jnp.pi * (2*x - 1)))
                
            return decay
        else:
            return 1.0
    
    def __call__(self, x: jax.Array, x_max: jax.Array, x_min: jax.Array):
        
        if self.scale == 'log':
            x, x_max, x_min = \
                jnp.log(jnp.maximum(x, self.eps)), jnp.log(jnp.maximum(x_min, self.eps)), jnp.log(x_max)
        
        x = (x - x_min) / (x_max - x_min) #### scale to 0, 1
        
        if self.clamp_input:
            x = self.soft_clamp(x)
        
        x = jnp.expand_dims(x, -1)
        decay = self.decay_fn(x)
        return jnp.concatenate([jnp.sin(self.npi * x), jnp.cos(self.npi * x)],
                               axis=-1) * decay