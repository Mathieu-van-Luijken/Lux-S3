import jax
import jax.numpy as jnp
from jax import random, grad, value_and_grad
import optax
import numpy as np
from typing import Any, Tuple

from ppo_agent import PPOAgent

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from src.luxai_s3.env import LuxAIS3Env
from src.luxai_s3.wrappers import LuxAIS3GymEnv
from src.luxai_runner.utils import to_json


def sample_action(key, move_probs, x_probs, y_probs):

    key, subkey = jax.random.split(key)
    move_action = jax.random.choice(
        subkey, move_probs.shape[-1], p=move_probs.flatten()
    )

    x_action = 0
    y_action = 0

    if move_action == 5:
        key, subkey_x, subkey_y = jax.random.split(key, 3)
        x_action = jax.random.choice(subkey_x, x_probs.shape[-1], p=x_probs.flatten())
        y_action = jax.random.choice(subkey_y, y_probs.shape[-1], p=y_probs.flatten())
    return jnp.array([move_action, x_action, y_action])


def init_agent(key):
    # Initialize the gameplay agent
    ppo_agent = PPOAgent()
    sample_positions = jax.numpy.zeros((2, 16, 2), dtype=jnp.int16)
    sample_energies = jax.numpy.zeros((2, 16), dtype=jnp.int16)
    sample_relics = jax.numpy.zeros((6, 2), dtype=jnp.int16)
    sample_tiles = jax.numpy.zeros((24, 24), dtype=jnp.int16)
    sample_energy = jax.numpy.zeros((24, 24), dtype=jnp.int16)
    sample_unit = jax.numpy.zeros((1, 3), dtype=jnp.int16)
    params = ppo_agent.init(
        key,
        sample_positions,
        sample_energies,
        sample_relics,
        sample_tiles,
        sample_energy,
        sample_unit,
    )
    return ppo_agent, params


def main(env, agent, params):
    num_episodes = 1000
    batch_size = 64
    optimizer = optax.adam(learning_rate=3e-4)
    opt_state = optimizer.init(params=params)

    for episode in range(1000):
        obs, info = env.reset()
        done = False

        while not done:
            action1, actions2, log_probs1, log_probs2 = [], [], [], []

            for player in [0, 1]:
                
                


if __name__ == "__main__":
    env = LuxAIS3GymEnv(numpy_output=True)
    key = jax.random.PRNGKey(2504)
    agent, params = init_agent(key=key)
    main(env, agent, params)
