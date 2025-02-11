import jax
import jax.numpy as jnp
from jax import random, grad, value_and_grad
import optax
import numpy as np
from typing import Any, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

from ppo_agent import PPOAgent
from loss import PPOLoss
from reward import calculate_reward
from utils import sample_action, calc_log_probs

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from src.luxai_s3.env import LuxAIS3Env
from src.luxai_s3.wrappers import LuxAIS3GymEnv
from src.luxai_runner.utils import to_json

from dataclasses import dataclass, field
import jax.numpy as jnp


@dataclass
class Trajectory:
    max_steps: int
    index: int = 0  # Tracks the current step

    unit_positions: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    player_unit_positions: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    unit_energies: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    relic_positions: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    tile_board: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    energy_board: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    actions: jnp.ndarray = field(default_factory=lambda: jnp.array([], dtype=jnp.int32))
    log_probs: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    rewards: jnp.ndarray = field(
        default_factory=lambda: jnp.array([], dtype=jnp.float32)
    )
    values: jnp.ndarray = field(
        default_factory=lambda: jnp.array([], dtype=jnp.float32)
    )

    def __post_init__(self):
        """Initialize arrays with fixed preallocated size."""
        self.unit_positions = jnp.zeros((self.max_steps, 2, 16, 2))
        self.player_unit_positions = jnp.zeros((self.max_steps, 16, 2))
        self.unit_energies = jnp.zeros((self.max_steps, 2, 16))
        self.relic_positions = jnp.zeros((self.max_steps, 6, 2))
        self.tile_board = jnp.zeros((self.max_steps, 24, 24))
        self.energy_board = jnp.zeros((self.max_steps, 24, 24))
        self.actions = jnp.zeros((self.max_steps, 16, 3))
        self.log_probs = jnp.zeros((self.max_steps, 16))
        self.rewards = jnp.zeros(self.max_steps, dtype=jnp.float32)
        self.values = jnp.zeros(self.max_steps, dtype=jnp.float32)

    def add_step(self, **kwargs):
        """
        Add new step data at the current index.
        Handles scalar values by converting them to arrays before concatenation.
        """
        if self.index >= self.max_steps:
            raise ValueError("Trajectory buffer is full!")

        for key, value in kwargs.items():
            if hasattr(self, key):
                current_attr = getattr(self, key)

                setattr(self, key, current_attr.at[self.index].set(value))
            else:
                raise ValueError(f"Invalid field name: {key}")
        self.index += 1  # Move to next index

    def reset(self):
        """Reset all fields to zero and reset index."""
        self.index = 0
        self.__post_init__()  # Reinitialize to zero


def init_agent(key):
    # Initialize the gameplay agent
    ppo_agent = PPOAgent()

    sample_player_unit_positions = jax.numpy.zeros((16, 2), dtype=jnp.int16)
    sample_positions = jax.numpy.zeros((2, 16, 2), dtype=jnp.int16)
    sample_energies = jax.numpy.zeros((2, 16), dtype=jnp.int16)
    sample_relics = jax.numpy.zeros((6, 2), dtype=jnp.int16)
    sample_tiles = jax.numpy.zeros((24, 24), dtype=jnp.int16)
    sample_energy = jax.numpy.zeros((24, 24), dtype=jnp.int16)
    params = ppo_agent.init(
        key,
        sample_positions,
        sample_player_unit_positions,
        sample_energies,
        sample_relics,
        sample_tiles,
        sample_energy,
    )
    return ppo_agent, params


def main(env, agent: PPOAgent, params, key):
    ppo_loss = PPOLoss()
    num_episodes = 1000
    nr_updates = 10
    update_step = 101

    optimizer = optax.adam(learning_rate=3e-4)
    opt_state = optimizer.init(params=params)
    pbar = tqdm(total=101)
    trajectory = Trajectory(max_steps=update_step)

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        trajectory.reset()
        episode_count = 0
        while not done:
            actions_dict = {}
            episode_count += 1
            for player in [0, 1]:
                if isinstance(obs, tuple):
                    obs = obs[0]
                if player == 1:
                    (
                        unit_positions,
                        player_unit_positions,
                        unit_energies,
                        relic_positions,
                        tile_board,
                        energy_board,
                    ) = agent.get_relevant_info(obs[f"player_{player}"], player)

                    value, move_probs = agent.apply(
                        params,
                        unit_positions,
                        player_unit_positions,
                        unit_energies,
                        relic_positions,
                        tile_board,
                        energy_board,
                    )

                    # Save the correct values
                    traj_actions = sample_action(
                        key=key,
                        move_probs=move_probs,
                        num_units=unit_positions[0].shape[0],
                    )
                    traj_log_probs = calc_log_probs(
                        move_probs=move_probs,
                        actions=traj_actions,
                        num_units=unit_positions[0].shape[0],
                    )
                    traj_value = value.item()
                else:
                    traj_actions = jnp.zeros((16, 3), dtype=jnp.int32)

                actions_dict[f"player_{player}"] = traj_actions

            (
                obs,
                state,
                _,
                _,
                _,
            ) = env.step(actions_dict)

            episode_reward += calculate_reward(obs["player_1"])

            # Update the trajectory:
            trajectory.add_step(
                unit_positions=unit_positions,
                player_unit_positions=unit_positions[
                    1
                ],  # TODO Unit positions are variable since we dont always have the same amount
                unit_energies=unit_energies,
                relic_positions=relic_positions,
                tile_board=tile_board,
                energy_board=energy_board,
                actions=traj_actions,
                log_probs=traj_log_probs,
                rewards=episode_reward,
                values=traj_value,
            )

            pbar.update(1)
            # Compute loss and update parameters after 101 steps
            if episode_count % update_step == 0:
                trajectory.advantages, trajectory.returns = (
                    ppo_loss.calculate_advantage(trajectory)
                )
                for i in range(nr_updates):
                    params, opt_state, batch_loss = ppo_loss(
                        params=params,
                        trajectory=trajectory,
                        agent=agent,
                        key=key,
                        opt_state=opt_state,
                    )
                # state = train_step(state, trajectory, agent, clip_eps)
                trajectory.reset()  # Reset trajectory for next batch


if __name__ == "__main__":
    env = LuxAIS3GymEnv(numpy_output=True)
    key = jax.random.PRNGKey(2504)
    agent, params = init_agent(key=key)
    main(env, agent, params, key)
