import jax
import jax.numpy as jnp
from jax import random, grad, value_and_grad
import optax
import numpy as np
from typing import Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from ppo_agent import PPOAgent
from loss import PPOLoss
from reward import calculate_reward
from utils import sample_action

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from src.luxai_s3.env import LuxAIS3Env
from src.luxai_s3.wrappers import LuxAIS3GymEnv
from src.luxai_runner.utils import to_json


@dataclass
class Trajectory:
    observations: list
    actions: list
    log_probs: list
    rewards: list
    values: list


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
    ppo_loss = PPOLoss()
    num_episodes = 1000
    batch_size = 64
    optimizer = optax.adam(learning_rate=3e-4)
    opt_state = optimizer.init(params=params)
    pbar = tqdm(total=100)
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        trajectory = Trajectory([], [], [], [], [])
        episode_count = 0

        while not done:
            actions_dict = {}
            episode_count += 1
            for player in [0, 1]:
                actions = []
                log_probs = []
                values = []
                if isinstance(obs, tuple):
                    obs = obs[0]
                if player == 1:
                    traj_obs = obs["player_1"]
                    (
                        unit_positions,
                        unit_energies,
                        relic_positions,
                        tile_board,
                        energy_board,
                    ) = agent.get_relevant_info(obs[f"player_{player}"])
                    traj_value = 0
                    for i, unit in enumerate(unit_positions[player]):
                        if jnp.any(unit[1] >= 0):
                            unit_info = jnp.append(unit, unit_energies[player][i])[
                                None, :
                            ]
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
                            log_probs.append(jnp.log(move_probs[0, action[0]]))
                            traj_value = value.item()
                        else:
                            actions.append([0, 0, 0])
                            log_probs.append(0)
                            values.append(0)

                    traj_actions = jnp.vstack(jnp.array(actions))
                    traj_log_probs = jnp.vstack(jnp.array(log_probs))

                else:
                    for i in range(16):
                        action = [0, 0, 0]
                        actions.append(action)

                actions_dict[f"player_{player}"] = jnp.vstack(jnp.array(actions))

            (
                obs,
                state,
                _,
                _,
                _,
            ) = env.step(actions_dict)

            episode_reward += calculate_reward(obs["player_1"])

            # Log all values
            trajectory.observations.append(traj_obs)
            trajectory.actions.append(traj_actions)
            trajectory.log_probs.append(traj_log_probs)
            trajectory.rewards.append(episode_reward)
            trajectory.values.append(traj_value)

            pbar.update(1)
            # Compute loss and update parameters after 101 steps
            if episode_count % 101 == 0:
                trajectory.advantages, trajectory.returns = (
                    ppo_loss.calculate_advantage(trajectory)
                )
                loss = ppo_loss(params, trajectory, agent)
                # state = train_step(state, trajectory, agent, clip_eps)
                trajectory = Trajectory(
                    [], [], [], [], []
                )  # Reset trajectory for next batch


if __name__ == "__main__":
    env = LuxAIS3GymEnv(numpy_output=True)
    key = jax.random.PRNGKey(2504)
    agent, params = init_agent(key=key)
    main(env, agent, params, key)
