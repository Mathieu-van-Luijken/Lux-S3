import jax
import jax.numpy as jnp
from jax import random, grad, value_and_grad
import optax
import numpy as np
from typing import Any, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

from ppo_agent import PPOAgent, preproces
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

    player_unit_positions: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    board_state: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    num_active_units: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    actions: jnp.ndarray = field(default_factory=lambda: jnp.array([], dtype=jnp.int32))
    log_probs: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    rewards: jnp.ndarray = field(
        default_factory=lambda: jnp.array([], dtype=jnp.float32)
    )
    values: jnp.ndarray = field(
        default_factory=lambda: jnp.array([], dtype=jnp.float32)
    )
    advantages: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    returns: jnp.ndarray = field(default_factory=lambda: jnp.array([]))

    def __post_init__(self):
        """Initialize arrays with fixed preallocated size."""
        self.player_unit_positions = jnp.zeros((self.max_steps, 16, 2), dtype=jnp.int16)
        self.board_state = jnp.zeros((self.max_steps, 1, 10, 24, 24), dtype=jnp.float32)
        self.num_active_units = jnp.zeros((self.max_steps, 1), dtype=jnp.int16)
        self.actions = jnp.zeros((self.max_steps, 16, 3), dtype=jnp.int16)
        self.log_probs = jnp.zeros((self.max_steps, 16), jnp.float32)
        self.rewards = jnp.zeros(self.max_steps, dtype=jnp.int32)
        self.values = jnp.zeros(self.max_steps, dtype=jnp.float32)
        self.advantages = jnp.zeros(self.max_steps, dtype=jnp.float32)
        self.returns = jnp.zeros(self.max_steps, dtype=jnp.float32)

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
    sample_board_state = jnp.zeros((1, 10, 24, 24), dtype=jnp.int16)
    params = ppo_agent.init(
        key,
        sample_player_unit_positions,
        sample_board_state,
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
                        num_active_units,
                    ) = agent.get_relevant_info(obs[f"player_{player}"], player)
                    board_state = preproces(
                        unit_positions=unit_positions,
                        unit_energies=unit_energies,
                        relic_positions=relic_positions,
                        tile_board=tile_board,
                        energy_board=energy_board,
                        player=player,
                    )
                    value, move_probs = agent.apply(
                        params,
                        board_state=board_state,
                        player_unit_positions=player_unit_positions,
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
                    traj_value = jnp.asarray(value.item(), dtype=jnp.float32)
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
                player_unit_positions=player_unit_positions,
                num_active_units=num_active_units,
                board_state=board_state,
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
