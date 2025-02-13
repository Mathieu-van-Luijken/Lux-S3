from lux.utils import direction_to
import sys
import numpy as np
from ppo_agent import PPOAgent, preproces, get_relevant_info
import jax.numpy as jnp
import jax
import flax.serialization
from jax.nn import sigmoid
from utils import sample_action


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

        key = jax.random.PRNGKey(2504)

        sample_player_unit_positions = jax.numpy.zeros((16, 2), dtype=jnp.int32)
        sample_board_state = jnp.zeros((1, 10, 24, 24), dtype=jnp.int32)
        self.params = self.ppo_agent.init(
            key,
            sample_player_unit_positions,
            sample_board_state,
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

        (
            unit_positions,
            player_unit_positions,
            unit_energies,
            relic_positions,
            tile_board,
            energy_board,
            num_active_units,
        ) = get_relevant_info(obs, self.player)

        board_state = preproces(
            unit_positions=unit_positions,
            unit_energies=unit_energies,
            relic_positions=relic_positions,
            tile_board=tile_board,
            energy_board=energy_board,
            player=self.player,
        )

        # Loop over all units and compute their actions
        actions = []
        value, move_probs = self.ppo_agent.apply(
            self.params,
            board_state=board_state,
            player_unit_positions=player_unit_positions,
        )
        actions = sample_action(key=self.key, move_probs=move_probs, num_units=16)
        return actions
