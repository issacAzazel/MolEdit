import jax 
import jax.numpy as jnp 
import flax
import flax.linen as nn
from typing import Tuple, Union, Callable

def shift(x):
    return jnp.concatenate([jnp.zeros_like(x[-1:]), 
                            x[:-1]], axis=0)

def concat_elu(x):
    return jax.nn.elu(jnp.concatenate([x, -x], axis=-1))

class ShiftedConv1D(nn.Module):
    out_channels: int
    kernel_size: Tuple[int]
    padding: Union[int, str] = 0
    
    @nn.compact
    def __call__(self, x):
        H = self.kernel_size[-1]
        
        x = jnp.pad(x, 
                    ((H-1, 0), (0, 0)),
                    mode='constant', constant_values=0)
        
        return nn.Conv(features=self.out_channels,
                       kernel_size=self.kernel_size,
                       padding=self.padding)(x) 
    
class GatedResidualLayer(nn.Module):
    conv: Union[nn.Module, Callable]
    activation: Callable
    n_channels: int
    kernel_size: Tuple[int]
    padding: Union[int, str] = 0
    shortcut_channels: bool=False
    
    @nn.compact
    def __call__(self, x, a=None):
        c1 = self.conv(
            self.n_channels, 
            self.kernel_size,
            padding=self.padding)(self.activation(x))
        if self.shortcut_channels:
            c1c = nn.Conv(
                self.n_channels, 
                kernel_size=(1,),
                padding=self.padding,
            )(self.activation(a))
            c1 = c1 + c1c
        c1 = self.activation(c1)
        c2 = self.conv(
            2*self.n_channels, 
            self.kernel_size,
            padding=self.padding)(c1)

        # L, C
        x1, x2 = c2[..., :self.n_channels], c2[..., self.n_channels:]
        
        out = x + x1 * jax.nn.sigmoid(x2)
        return out

def causal_attention(k, q, v, mask, nh):
    L, dq = q.shape
    _, dv = v.shape

    k, q, v = k.T, q.T, v.T
    flat_q = q.reshape(nh, dq//nh, L) * (dq // nh) ** (-0.5)
    flat_k = k.reshape(nh, dq//nh, L)
    flat_v = v.reshape(nh, dv//nh, L)

    logits = jnp.matmul(jnp.transpose(flat_q, (0, 2, 1)), flat_k)
    
    logits = jnp.where(mask, logits, -1e10)
    weights = jax.nn.softmax(logits, -1)
    weights = jnp.where(mask, weights, 0.0)

    attn_out = jnp.matmul(weights, jnp.transpose(flat_v, (0, 2, 1)))
    attn_out = jnp.transpose(attn_out, (1, 0, 2))

    return attn_out.reshape(L, -1)

class AttentionGatedResidualBlock(nn.Module):
    n_channels: int
    n_res_layers: int 
    nh: int
    dq: int 
    dv: int 
    
    @nn.compact 
    def __call__(self, x, attn_mask):
        ul = x
        for _ in range(self.n_res_layers):
            input_gated_resnet = GatedResidualLayer(ShiftedConv1D, concat_elu, self.n_channels, (2,))
            ul = input_gated_resnet(ul)

        kv = GatedResidualLayer(nn.Conv, concat_elu, 
                                2*self.n_channels, (1,))(jnp.concatenate([x, ul], axis=-1))
        kv = nn.Conv(self.dq + self.dv, (1,))(kv)
        k, v = kv[..., :self.dq], kv[..., self.dq:]

        q = GatedResidualLayer(nn.Conv, concat_elu, 
                               self.n_channels, (1,))(ul)
        q = nn.Conv(self.dq, (1,))(q)

        attn_out = causal_attention(k, q, v, attn_mask, self.nh)
        
        return GatedResidualLayer(nn.Conv, concat_elu, self.n_channels, (1,), 
                                  shortcut_channels=True)(ul, attn_out)
        

class SNAIL(nn.Module):
    out_dims: int 
    n_channels: int=128
    n_res_layers: int=5
    n_attn_layers: int=12
    attn_nh: int=1
    attn_dq: int=16
    attn_dv: int=128
    
    @nn.compact
    def __call__(self, x):
        attn_mask = jnp.tril(
            jnp.ones((1, x.shape[0], x.shape[0]), 
                     dtype=jnp.bool_), -1)
        x = jnp.pad(x[..., None], ((0,0), (0,1)), constant_values=1)
        ul_input_d = ShiftedConv1D(self.n_channels, 
                                   kernel_size=(3,))
        ul = shift(ul_input_d(x))
        
        for _ in range(self.n_attn_layers):
            ul = AttentionGatedResidualBlock(
                self.n_channels, 
                self.n_res_layers,
                self.attn_nh,
                self.attn_dq,
                self.attn_dv)(ul, attn_mask)
        
        return nn.Conv(
            self.out_dims, 
            kernel_size=(1,))(jax.nn.elu(ul))