import jax
import jax.numpy as jnp
from jax import random, grad, value_and_grad
import numpy as np
from typing import Any, Tuple
from ...src.luxai_s3.env import LuxAIS3Env

class PPOAgent:
    def __init__(self, obs_dim, act_dim, hidden_sizes, key):
        self.act_dim = act_dim
        key, subkey = random.split(key)
        self.policy_params = self.create_network([obs_dim] + hidden_sizes + [act_dim], subkey)

        key, subkey = random.split(key)
        self.value_params = self.create_network([obs_dim] + hidden_sizes + [1], subkey)

    def create_network(self, layer_sizes, key):
        params=[]
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            key, subkey= random.split(key)
            w =  random.normal(subkey, (in_size, out_size)) * jnp.sqrt(2.0/in_size)
            b = jnp.zeros(out_size)
            params.append((w,b))
        
        return params
    
    def policy_network(self, params, state):
        for w, b in params[:-1]:
            state = jax.nn.relu(jnp.dot(state, w)+b)
        final_w, final_b = params[-1]
        logits = jnp.dot(state, final_w) + final_b

        return logits

    def value_network(self, params, state): 
        for w, b in params[-1]:
            state = jax.nn.relu(jnp.dot(state, w)+ b)
            final_w, final_b = params[-1]
            value = jnp.dot(state, final_w) + final_b
        return value.squeeze()
    