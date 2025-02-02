import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random, grad, value_and_grad
import numpy as np
from typing import Any, Tuple
from ...src.luxai_s3.env import LuxAIS3Env


class PPOLoss(nn.Module):
    epsilon: float = 0.2
    c1: float = 0.5
    c2: float = 0.01

    def __call__(self, params, trajectory, agent):
        observations, actions, old_log_probs, advantages, returns = (
            trajectory.observations,
            trajectory.actions,
            trajectory.log_probs,
            trajectory.advantages,
            trajectory.returns,
        )

        _ new_log_probs 
