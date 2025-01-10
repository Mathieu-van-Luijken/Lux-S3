from lux.utils import direction_to
import sys
import numpy as np
from ppo_agent import PPOAgent
import jax.numpy as jnp
import jax
import flax.serialization
from jax.nn import sigmoid


class Agent:
    def __init__(
        self,
        player: str,
        load_parameters: bool = False,
        filename: str = None,
    ) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)

        # Initialize the gameplay agent
        self.ppo_agent = PPOAgent()
        self.rng = jax.random.PRNGKey(2504)
        x = jax.random.normal(self.rng, (1, 8, 24, 24))
        self.params = self.ppo_agent.init(self.rng, x)

        # # Load parameters
        # if load_parameters:
        #     with open(filename, "rb") as f:
        #         self.params = flax.serialization.from_bytes(self.params, f.read())

    def preproces(
        self,
        unit_positions,
        unit_energies,
        relic_positions,
        tile_board,
        energy_board,
    ):
        """Generates the board state tensor."""
        # Create unit board
        unit_board = jnp.zeros((24, 24), dtype=int)
        for unit_pos in unit_positions:
            if unit_pos[0] >= 0:
                unit_board = unit_board.at[unit_pos[0], unit_pos[1]].add(1)

        # Create relic board
        relic_board = jnp.zeros((24, 24), dtype=int)
        for relic_pos in relic_positions:
            if relic_pos[0] >= 0:
                relic_board = relic_board.at[relic_pos[0], relic_pos[1]].add(1)

        # Obtain tile information
        vision_board = jnp.where(tile_board == -1, 0, 1)
        normal_tiles = jnp.where(tile_board == 0, 1, 0)
        nebula_tiles = jnp.where(tile_board == 1, 1, 0)
        asteroid_tiles = jnp.where(tile_board == 2, 1, 0)

        # Energy
        unit_energy_board = jnp.zeros((24, 24), dtype=float)
        for i, unit_pos in enumerate(unit_positions):
            if unit_pos[0] >= 0:
                unit_energy_board = unit_energy_board.at[unit_pos[0], unit_pos[1]].add(
                    unit_energies[i].item()
                )

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
            ]
        )
        return board_state_tensor[None, :, :, :]

    def train_act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        # Concatenate unit and energy per unit info
        units_pos = jnp.array(obs["units"]["position"][self.team_id])
        units_energy = (jnp.array(obs["units"]["energy"][self.team_id]) / 100)[:, None]
        # unit_info = jnp.concat([unit_positions, unit_energys], axis=-1)

        # Obtain Relic positions
        relic_positions = jnp.array(obs["relic_nodes"])

        tile_board = jnp.array(obs["map_features"]["tile_type"])
        energy_board = jnp.array(obs["map_features"]["energy"])

        board_state_tensor = self.preproces(
            units_pos=units_pos,
            units_energy=units_energy,
            relic_positions=relic_positions,
            tile_board=tile_board,
            energy_board=energy_board,
        )

        actions = []
        for unit in units_pos:
            value, logits, x_coord, y_coord = self.ppo_agent.apply(
                self.params, board_state_tensor
            )

        return actions

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        # Concatenate unit and energy per unit info
        unit_positions = jnp.array(obs["units"]["position"][self.team_id])
        unit_energies = (jnp.array(obs["units"]["energy"][self.team_id]) / 100)[:, None]

        # Obtain Relic positions
        relic_positions = jnp.array(obs["relic_nodes"])

        # Get the board energies
        tile_board = jnp.array(obs["map_features"]["tile_type"])
        energy_board = jnp.array(obs["map_features"]["energy"])

        board_state_tensor = self.preproces(
            unit_positions=unit_positions,
            unit_energies=unit_energies,
            relic_positions=relic_positions,
            tile_board=tile_board,
            energy_board=energy_board,
        )
        actions = []
        for unit in unit_positions:
            value, logits, x_coord, y_coord = self.ppo_agent.apply(
                self.params, board_state_tensor
            )
            action = jnp.argmax(sigmoid(logits))
            x_coord = jnp.argmax(sigmoid(x_coord))
            y_coord = jnp.argmax(sigmoid(y_coord))
            actions.append([action, x_coord, y_coord])

        return jnp.vstack(actions)
