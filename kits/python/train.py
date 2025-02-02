import os
import sys
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import grad, random, value_and_grad
from ppo_agent import PPOAgent

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../src")
from reward import calculate_reward

from luxai_runner.utils import to_json
from luxai_s3.env import LuxAIS3Env
from luxai_s3.wrappers import LuxAIS3GymEnv


def sample_action(key, move_probs):

    key, subkey = jax.random.split(key)
    move_action = jax.random.choice(
        subkey, move_probs.shape[-1], p=move_probs.flatten()
    )

    if move_action == 5:
        key, subkey_x, subkey_y = jax.random.split(key, 3)
        move_action = 0

    return jnp.array([move_action, 0, 0])


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


def main(env, agent: PPOAgent, params, key):
    num_episodes = 1000
    batch_size = 64
    optimizer = optax.adam(learning_rate=3e-4)
    opt_state = optimizer.init(params=params)

    for episode in range(505):
        obs, info = env.reset()
        done = False

        while not done:
            action1, actions2, log_probs1, log_probs2 = [], [], [], []
            actions_dict = {}
            for player in [0, 1]:
                #     if player == 0:
                #         action = np.random.choice([2, 3])
                #         action_state = np.array([action, 0, 0])
                #         player_1_actions = jnp.stack([action_state] * 16, axis=0)
                #     if player == 1:
                #         action = np.random.choice([1, 4])
                #         action_state = np.array([action, 0, 0])
                #         player_2_actions = jnp.stack([action_state] * 16, axis=0)
                # final_action = {"player_0": player_1_actions, "player_1": player_2_actions}
                # obs = env.step(final_action)
                actions = []
                log_prob = []
                if isinstance(obs, tuple):
                    obs = obs[0]
                (
                    unit_positions,
                    unit_energies,
                    relic_positions,
                    tile_board,
                    energy_board,
                ) = agent.get_relevant_info(obs[f"player_{player}"])
                for i, unit in enumerate(unit_positions[player]):
                    if jnp.any(unit[1] >= 0):
                        unit_info = jnp.append(unit, unit_energies[player][i])[None, :]
                        value, move_probs = agent.apply(
                            params,
                            unit_positions,
                            unit_energies,
                            relic_positions,
                            tile_board,
                            energy_board,
                            unit_info,
                        )
                        action = sample_action(key=key, move_probs=move_probs)
                        actions.append(action)
                    else:
                        actions.append([0, 0, 0])
                actions_dict[f"player_{player}"] = jnp.vstack(actions)
            obs = env.step(actions_dict)


if __name__ == "__main__":
    env = LuxAIS3GymEnv(numpy_output=True)
    key = jax.random.PRNGKey(2504)
    agent, params = init_agent(key=key)
    main(env, agent, params, key)
