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
        init_x = jax.random.normal(self.rng, (1, 8, 24, 24))
        init_info = jnp.array([[0, 0, 1]])
        self.params = self.ppo_agent.init(self.rng, init_x, init_info)

        # # Load parameters
        # if load_parameters:
        #     with open(filename, "rb") as f:
        #         self.params = flax.serialization.from_bytes(self.params, f.read())

    def preproces(
        self,
        unit_positions: jax.Array,
        unit_energies: jax.Array,
        relic_positions: jax.Array,
        tile_board: jax.Array,
        energy_board: jax.Array,
    ):
        """Generates the board state tensor."""
        # Create unit board
        unit_board = jnp.zeros((24, 24), dtype=int)
        existing_units = unit_positions[:, 0] >= 0
        existing_unit_pos = unit_positions[existing_units]
        x_coords, y_coords = existing_unit_pos[:, 0], existing_unit_pos[:, 1]
        unit_board = unit_board.at[x_coords, y_coords].add(1)

        # Create the energy board for units
        unit_energy_board = jnp.zeros((24, 24), dtype=float)
        existing_energies = unit_energies[existing_units].flatten()
        unit_energy_board = unit_energy_board.at[x_coords, y_coords].add(
            existing_energies
        )

        # Create relic board
        relic_board = jnp.zeros((24, 24), dtype=int)
        found_relic = relic_positions[:, 0] >= 0
        found_relic_pos = relic_positions[found_relic]
        relic_x_coords, relic_y_coords = found_relic_pos[:, 0], found_relic_pos[:, 1]
        relic_board = relic_board.at[relic_x_coords, relic_y_coords].add(1)

        # Obtain the vision board
        vision_board = jnp.where(tile_board == -1, 0, 1)

        # Use onehot encoding to find locations of other tiles
        tile_types = jax.nn.one_hot(tile_board, num_classes=4, dtype=int)
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
            ]
        )
        return board_state_tensor[None, :, :, :]

    def train_act(self, step: int, obs, remainingOverageTime: int = 60):
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

        # Preprocess the observation into a tensor of stacked boards
        board_state_tensor = self.preproces(
            unit_positions=unit_positions,
            unit_energies=unit_energies,
            relic_positions=relic_positions,
            tile_board=tile_board,
            energy_board=energy_board,
        )

        # Loop over all units and compute their actions
        actions = []
        for i, unit in enumerate(unit_positions):
            if unit[1] >= 0:
                unit_info = jnp.concat([unit, unit_energies[i]])[None, :]
                value, logits, x_coord, y_coord = self.ppo_agent.apply(
                    self.params, board_state_tensor, unit_info
                )
                action = jnp.argmax(softmax(logits))
                x_coord = jnp.argmax(softmax(x_coord))
                y_coord = jnp.argmax(softmax(y_coord))
                actions.append([action, x_coord, y_coord])
            else:
                actions.append([0, 0, 0])

        return jnp.vstack(actions)
