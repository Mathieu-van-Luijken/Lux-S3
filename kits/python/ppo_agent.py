import jax
import flax.linen as nn
import jax.numpy as jnp
from jax import random, grad, value_and_grad
from flax.training import train_state
import optax
import numpy as np
from typing import Any, Tuple

class PPOAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, key):
        self.act_dim = act_dim
        key, subkey = random.split(key)

    