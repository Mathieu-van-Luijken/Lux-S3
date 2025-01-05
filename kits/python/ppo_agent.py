import jax
import flax.linen as nn
import jax.numpy as jnp
from jax import random, grad, value_and_grad
from flax.training import train_state
import optax
import numpy as np
from typing import Any, Tuple


class PPOAgent(nn.Module):
    def __init__(self, player: str) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        # self.env_cfg = env_cfg

        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()

    def create_board_tensor(self, obs):
        obs = obs[self.player]
        # Create unit board
        unit_board = np.zeros((24, 24), dtype=int)
        for unit_pos in np.array(obs["units"]["position"][self.team_id]):
            if unit_pos[0] >= 0:
                unit_board[unit_pos[0], unit_pos[1]] += 1

        # Create relic board
        relic_board = np.zeros((24, 24), dtype=int)
        for relic_pos in np.array(obs["relic_nodes"]):
            if relic_pos[0] >= 0:
                relic_board[relic_pos[0], relic_pos[1]] += 1

        # Obtain tile information
        tile_board = jnp.array(obs["map_features"]["tile_type"])

        # Create tile boards board
        vision_board = jnp.where(tile_board == -1, 0, 1)
        normal_tiles = jnp.where(tile_board == 0, 1, 0)
        nebula_tiles = jnp.where(tile_board == 1, 1, 0)
        asteroid_tiles = jnp.where(tile_board == 2, 1, 0)

        # Obtain energy tiles
        energy_tiles = obs["map_features"]["energy"]

        # Energy
        energy = jnp.array(obs["units"]["energy"] / 100)[0][:, None]  # NOTE fix the 0
        unit_positions = np.array(obs["units"]["position"])[0]
        unit_energy = jnp.concat([energy, unit_positions], axis=-1)
        unit_energy_board = np.zeros((24, 24), dtype=int)
        for energy_pos in np.array(unit_energy):
            if energy_pos[0] >= 0:
                unit_energy_board[energy[0], energy_pos[1]] += energy_pos[2]

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
                energy_tiles,
            ]
        )

        return board_state_tensor

    def act(self, step: int, obs, remainingOvergaeTime: int = 60):
        board_state_tensor = self.create_board_tensor(obs)
