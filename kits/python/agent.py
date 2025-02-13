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
        self.rng = jax.random.PRNGKey(2504)
        self.ppo_agent = PPOAgent(self.opp_player)
        sample_positions = jax.numpy.zeros((2, 16, 2), dtype=jnp.int32)
        sample_energies = jax.numpy.zeros((2, 16), dtype=jnp.int32)
        sample_relics = jax.numpy.zeros((6, 2), dtype=jnp.int32)
        sample_tiles = jax.numpy.zeros((24, 24), dtype=jnp.int32)
        sample_energy = jax.numpy.zeros((24, 24), dtype=jnp.int32)
        sample_unit = jax.numpy.zeros((1, 3), dtype=jnp.int32)
        self.params = self.ppo_agent.init(
            self.rng,
            sample_positions,
            sample_energies,
            sample_relics,
            sample_tiles,
            sample_energy,
            sample_unit,
        )

        # # Load parameters
        # if load_parameters:
        #     with open(filename, "rb") as f:
        #         self.params = flax.serialization.from_bytes(self.params, f.read())

    def sample_action(self, move_probs, x_probs, y_probs):

        key, subkey = jax.random.split(self.rng)
        move_action = jax.random.choice(
            subkey, move_probs.shape[-1], p=move_probs.flatten()
        )

        x_action = 0
        y_action = 0

        if move_action == 5:
            key, subkey_x, subkey_y = jax.random.split(key, 3)
            x_action = jax.random.choice(
                subkey_x, x_probs.shape[-1], p=x_probs.flatten()
            )
            y_action = jax.random.choice(
                subkey_y, y_probs.shape[-1], p=y_probs.flatten()
            )
        return jnp.array([move_action, x_action, y_action])

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """

        unit_positions, unit_energies, relic_positions, tile_board, energy_board = (
            self.ppo_agent.get_relevant_info(obs=obs)
        )

        # Loop over all units and compute their actions
        actions = []
        for i, unit in enumerate(unit_positions[self.team_id]):
            if jnp.any(unit[1] >= 0):
                unit_info = jnp.append(unit, unit_energies[self.team_id][i])[None, :]
                value, move_probs, x_probs, y_probs = self.ppo_agent.apply(
                    self.params,
                    unit_positions,
                    unit_energies,
                    relic_positions,
                    tile_board,
                    energy_board,
                    unit_info,
                )
                action = self.ppo_agent.sample_action(move_probs, x_probs, y_probs)
                actions.append(action)
            else:
                actions.append([0, 0, 0])

        return jnp.vstack(actions)
