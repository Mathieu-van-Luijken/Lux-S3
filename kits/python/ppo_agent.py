import jax
import flax.linen as nn
import jax.numpy as jnp
from jax import random, grad, value_and_grad
from flax.training import train_state
import optax
import numpy as np
from typing import Any, Tuple
from functools import partial
from jax import lax, jit


@jit
def scatter_update(board, coords, values, valid_mask):
    """
    For each row in `coords` (shape [N, 2]), if valid_mask[i] is True, we use coords[i],
    otherwise we substitute a dummy coordinate (here (0,0)). In parallel, we zero out the update
    for invalid entries. Then we do a scatter-add.
    """
    # Use safe_coords: if valid, use the coordinate; if not, use (0, 0)
    safe_coords = jnp.where(valid_mask[:, None], coords, jnp.zeros_like(coords))
    # For invalid entries, update value becomes 0.
    updates = jnp.where(valid_mask, values, 0)
    return board.at[safe_coords[:, 0], safe_coords[:, 1]].add(updates)


@jit
def preproces(
    unit_positions,
    unit_energies,
    relic_positions,
    tile_board,
    energy_board,
    player,
):
    # Create a friendly unit board
    friendly_unit_board = jnp.zeros((24, 24), dtype=jnp.float32)
    friendly_units = unit_positions[player]
    friendly_valid = jnp.logical_not(jnp.all(friendly_units == -1, axis=-1))
    friendly_unit_board = scatter_update(
        friendly_unit_board,
        friendly_units,
        jnp.ones(friendly_units.shape[0], dtype=jnp.float32),
        friendly_valid,
    )

    # Create a friendly unit board
    enemy_unit_board = jnp.zeros((24, 24), dtype=jnp.float32)

    # Obtain all coords of existing friendly units
    enemy_units = unit_positions[1 - player]
    enemy_valid = jnp.logical_not(jnp.all(enemy_units == -1, axis=-1))
    enemy_unit_board = scatter_update(
        enemy_unit_board,
        enemy_units,
        jnp.ones(enemy_units.shape[0], dtype=jnp.float32),
        enemy_valid,
    )

    # Create the energy board for units
    friendly_energy_board = jnp.zeros((24, 24), dtype=jnp.float32)
    safe_friendly_energies = jnp.where(friendly_valid, unit_energies[0].flatten(), 0)
    friendly_energy_board = scatter_update(
        friendly_energy_board,
        friendly_units,
        safe_friendly_energies,
        friendly_valid,
    )

    # Create enemy energy board for units
    enemy_energy_board = jnp.zeros((24, 24), dtype=jnp.float32)
    safe_enemy_energies = jnp.where(enemy_valid, unit_energies[1].flatten(), 0)
    enemy_energy_board = scatter_update(
        enemy_energy_board, enemy_units, safe_enemy_energies, enemy_valid
    )

    # Create relic board
    relic_board = jnp.zeros((24, 24), dtype=jnp.float32)

    relic_valid = jnp.logical_not(jnp.all(relic_positions == -1, axis=-1))
    relic_board = scatter_update(
        relic_board,
        relic_positions,
        jnp.ones(relic_positions.shape[0], dtype=jnp.float32),
        relic_valid,
    )

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
            friendly_unit_board,
            enemy_unit_board,
            friendly_energy_board,
            enemy_energy_board,
            relic_board,
            vision_board,
            normal_tiles,
            nebula_tiles,
            asteroid_tiles,
            energy_board,
        ],
    )
    return board_state_tensor[None, ...]


class PPOAgent(nn.Module):

    def get_relevant_info(self, obs, player):
        # Concatenate unit and energy per unit info
        player_unit_positions = jnp.array(obs["units"]["position"][player])

        unit_positions = jnp.array(obs["units"]["position"])
        unit_energies = jnp.array(obs["units"]["energy"]) / 100

        # Obtain Relic positions
        relic_positions = jnp.array(obs["relic_nodes"])

        # Get the board energies
        tile_board = jnp.array(obs["map_features"]["tile_type"])
        energy_board = jnp.array(obs["map_features"]["energy"])

        # num active units:
        valid_mask = player_unit_positions[:, 0] != -1
        num_active_units = jnp.asarray(
            player_unit_positions[valid_mask].shape[0], dtype=jnp.int16
        )

        return (
            unit_positions,
            player_unit_positions,
            unit_energies,
            relic_positions,
            tile_board,
            energy_board,
            num_active_units,
        )

    @nn.compact
    def cnn_embedder(self, board_state_tensor):
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
        # Somehow need to input the correct unit positions
        return board_embedding

    @nn.compact
    def critic(self, embedding):
        """Computes the value estimation."""
        x = nn.Dense(features=64)(embedding)
        x = nn.relu(x)
        value = nn.Dense(features=1)(x)
        return value

    @nn.compact
    def actor(self, embedding, player_unit_positions):
        board_embedding_broadcasted = jnp.broadcast_to(
            embedding, (player_unit_positions.shape[0], 128)
        )
        embedding = jnp.concat(
            [board_embedding_broadcasted, player_unit_positions], axis=-1
        )
        """Computes the action logits."""
        x = nn.Dense(features=64)(embedding)
        x = nn.relu(x)
        action_logits = nn.Dense(features=6)(x)
        return nn.softmax(action_logits)

    @nn.compact
    def __call__(self, player_unit_positions, board_state):
        """Forward pass of the PPO agent."""
        embedding = self.cnn_embedder(board_state)
        value = self.critic(embedding)
        action_probs = self.actor(embedding, player_unit_positions)
        return value, action_probs
