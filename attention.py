from jax import numpy as jnp
from jax import random
from jax import vmap
from jax import jit
from jax import grad
from jax import value_and_grad
import einops

def softmax(x):
    return jnp.exp(x) / jnp.sum(jnp.exp(x))

def identity(x):
    return x

def G(x, W_Q, W_K, scale=-1/2):
    # x: [b s emb]
    # W_Q, W_K: [emb d] (for single head when used with vmap)
    Q = jnp.einsum("b s e, e d -> b s d", x, W_Q)
    K = jnp.einsum("b s e, e d -> b s d", x, W_K)
    # Correct einsum for attention scores
    return jnp.einsum("b s d, b t d -> b s t", Q, K) / jnp.sqrt(Q.shape[2])**scale

def f(x, W_Q, W_K, W_V, xi, scale=-1/2):
    attn = xi(G(x, W_Q, W_K, scale=scale))
    V = jnp.einsum("b s e, e d -> b s d", x, W_V)
    return jnp.einsum("b s t, b t d -> b s d", attn, V)

def mha(x, W_Q, W_K, W_V, W_O, xi, scale=-1/2):
    """
    x: b s emb
    W_Q, W_K, W_V: h emb d
    W_O: h*d d
    xi: s s -> s
    scale: float

    return: b s d
    """
    
    # Apply f for each head
    heads = vmap(lambda w_q, w_k, w_v: f(x, w_q, w_k, w_v, xi, scale), in_axes=(0, 0, 0))(W_Q, W_K, W_V)
    # heads shape is now [h b s d]
    
    # Rearrange to put batch first and combine heads
    heads = einops.rearrange(heads, "h b s d -> b s (h d)")
    
    # Project back to desired dimension
    return jnp.einsum("b s t, t d -> b s d", heads, W_O)

