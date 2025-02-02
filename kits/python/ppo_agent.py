import jax
import flax.linen as nn
import jax.numpy as jnp
from jax import random, grad, value_and_grad
from flax.training import train_state
import optax
import numpy as np
from typing import Any, Tuple
from functools import partial


class PPOAgent(nn.Module):

    def preproces(
        self,
        unit_positions,
        unit_energies,
        relic_positions,
        tile_board,
        energy_board,
    ):

        # Create unit board
        unit_board = jnp.zeros((24, 24), dtype=jnp.float32)

        # Obtain all coords of existing friendly units
        friendly_units = unit_positions[0]
        friendly_valid = jnp.logical_not(jnp.all(friendly_units == -1, axis=-1))
        friendly_coords = friendly_units[friendly_valid]

        # Obtain all coords of existing enemy units
        enemy_units = unit_positions[1]
        enemy_valid = jnp.logical_not(jnp.all(enemy_units == -1, axis=-1))
        enemy_coords = enemy_units[enemy_valid]

        # Set all positions on the board
        unit_board = unit_board.at[friendly_coords[:, 0], friendly_coords[:, 1]].set(
            1
        )  # TODO make additive
        unit_board = unit_board.at[enemy_coords[:, 0], enemy_coords[:, 1]].set(-1)

        # Create the energy board for units
        unit_energy_board = jnp.zeros((24, 24), dtype=jnp.float32)

        # Set all friendly energies
        friendly_energies = unit_energies[0][friendly_valid].flatten()
        unit_energy_board = unit_energy_board.at[
            friendly_coords[:, 0], friendly_coords[:, 1]
        ].add(friendly_energies)

        # Set all enemy energies
        enemy_energies = unit_energies[0][enemy_valid].flatten()
        unit_energy_board = unit_energy_board.at[
            enemy_coords[:, 0], enemy_coords[:, 1]
        ].add(enemy_energies)

        # Create relic board
        relic_board = jnp.zeros((24, 24), dtype=jnp.float32)
        found_relic = relic_positions[:, 0] >= 0
        found_relic_pos = relic_positions[found_relic]
        relic_x_coords, relic_y_coords = found_relic_pos[:, 0], found_relic_pos[:, 1]
        relic_board = relic_board.at[relic_x_coords, relic_y_coords].add(1)

        # Obtain the vision board
        vision_board = jnp.where(tile_board == -1, 0, 1)

        # Use onehot encoding to find locations of other tiles
        tile_types = jax.nn.one_hot(tile_board, num_classes=4, dtype=jnp.float32)
        normal_tiles = tile_types[..., 0]
        nebula_tiles = tile_types[..., 1]
        asteroid_tiles = tile_types[..., 2]

        # Stack all boards
        board_state_tensor = jnp.stack(
            [
                unit_board,
                unit_energy_board,
                relic_board,
                vision_board,
                normal_tiles,
                nebula_tiles,
                asteroid_tiles,
                energy_board,
            ],
        )
        return board_state_tensor[None, ...]

    def get_relevant_info(self, obs):
        # Concatenate unit and energy per unit info
        unit_positions = jnp.array(obs["units"]["position"])
        unit_energies = jnp.array(obs["units"]["energy"]) / 100

        # Obtain Relic positions
        relic_positions = jnp.array(obs["relic_nodes"])

        # Get the board energies
        tile_board = jnp.array(obs["map_features"]["tile_type"])
        energy_board = jnp.array(obs["map_features"]["energy"])

        return (
            unit_positions,
            unit_energies,
            relic_positions,
            tile_board,
            energy_board,
        )

    @nn.compact
    def cnn_embedder(self, board_state_tensor, unit_info):
        """Embeds the board state tensor into a 128-dimensional space."""
        x = board_state_tensor.transpose(0, 2, 3, 1)  # Add channel dimension
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=128)(x)
        board_embedding = nn.relu(x)
        embedding = jnp.concat([board_embedding, unit_info], axis=-1)
        return embedding

    @nn.compact
    def critic(self, embedding):
        """Computes the value estimation."""
        x = nn.Dense(features=64)(embedding)
        x = nn.relu(x)
        value = nn.Dense(features=1)(x)
        return value

    @nn.compact
    def actor(self, embedding):
        """Computes the action logits."""
        x = nn.Dense(features=64)(embedding)
        x = nn.relu(x)
        action_logits = nn.Dense(features=6)(x)
        return nn.softmax(action_logits)

    @nn.compact
    def __call__(
        self,
        unit_positions,
        unit_energies,
        relic_positions,
        tile_board,
        energy_board,
        unit_info,
    ):
        """Forward pass of the PPO agent."""
        board_state_tensor = self.preproces(
            unit_positions, unit_energies, relic_positions, tile_board, energy_board
        )
        embedding = self.cnn_embedder(board_state_tensor, unit_info=unit_info)
        value = self.critic(embedding)
        action_probs = self.actor(embedding)
        return value, action_probs
