import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random, grad, value_and_grad
import random
import numpy as np
from typing import Any, Tuple
from utils import dynamic_slice
import optax


class PPOLoss(nn.Module):
    epsilon: float = 0.2
    gamma: float = 0.99
    lambda_: float = 0.95
    c1: float = 0.5
    c2: float = 0.01
    minibatch_size: int = 10
    learning_rate: float = 0.001

    def setup(self):
        self.optimizer = optax.adam(self.learning_rate)
        self.key = jax.random.PRNGKey(1480928)

    def __call__(self, params, trajectory, agent, key, opt_state):
        (
            unit_positions,
            player_unit_positions,
            unit_energies,
            relic_positions,
            tile_board,
            energy_board,
            actions,
            log_probs,
            advantages,
            returns,
        ) = (
            trajectory.unit_positions,
            trajectory.player_unit_positions,
            trajectory.unit_energies,
            trajectory.relic_positions,
            trajectory.tile_board,
            trajectory.energy_board,
            trajectory.actions,
            trajectory.log_probs,
            trajectory.advantages,
            trajectory.returns,
        )

        num_samples = unit_positions.shape[0]
        perm = jax.random.permutation(key, num_samples)

        # Shuffle the data to obtain random batches
        shuffled_unit_positions = unit_positions[perm]
        shuffled_player_unit_positions = player_unit_positions[perm]
        shuffled_unit_energies = unit_energies[perm]
        shuffled_relic_positions = relic_positions[perm]
        shuffled_tile_board = tile_board[perm]
        shuffled_energy_board = energy_board[perm]
        shuffled_actions = actions[perm]
        shuffled_log_probs = log_probs[perm]
        shuffled_advantages = advantages[perm]
        shuffled_returns = returns[perm]

        def batch_update(carry, batch_idx):
            params, opt_state = carry

            (
                unit_positions_batch,
                player_unit_positions_batch,
                unit_energies_batch,
                relic_positions_batch,
                tile_board_batch,
                energy_board_batch,
                actions_batch,
                old_log_probs_batch,
                advantages_batch,
                returns_batch,
            ) = dynamic_slice(
                unit_positions=shuffled_unit_positions,
                player_unit_positions=shuffled_player_unit_positions,
                unit_energies=shuffled_unit_energies,
                relic_positions=shuffled_relic_positions,
                tile_board=shuffled_tile_board,
                energy_board=shuffled_energy_board,
                actions=shuffled_actions,
                old_log_probs=shuffled_log_probs,
                advantages=shuffled_advantages,
                returns=shuffled_returns,
                minibatch_size=self.minibatch_size,
                batch_idx=batch_idx,
            )

            loss, grads = value_and_grad(self.loss_fn)(
                params,
                agent,
                unit_positions_batch,
                player_unit_positions_batch,
                unit_energies_batch,
                relic_positions_batch,
                tile_board_batch,
                energy_board_batch,
                actions_batch,
                old_log_probs_batch,
                advantages_batch,
                returns_batch,
            )
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (params, opt_state), losses = jax.lax.scan(
            lambda carry, batch_idx: batch_update(carry, batch_idx),
            (params, opt_state),
            jnp.arange(0, num_samples, self.minibatch_size),
        )
        return params, opt_state, jnp.mean(losses)

    def loss_fn(
        self,
        params,
        agent,
        unit_positions_batch,
        player_unit_positions_batch,
        unit_energies_batch,
        relic_positions_batch,
        tile_board_batch,
        energy_board_batch,
        actions_batch,
        old_log_probs_batch,
        advantages_batch,
        returns_batch,
    ):

        value, move_probs = agent.apply(
            params,
            unit_positions_batch,
            player_unit_positions_batch,
            unit_energies_batch,
            relic_positions_batch,
            tile_board_batch,
            energy_board_batch,
        )

        # Make the lists into a jnp array for easier computation
        new_log_probs = jnp.vstack(np.array(new_log_probs))
        old_log_probs = jnp.vstack(np.array(old_log_probs_batch))
        new_values = jnp.vstack(np.array(new_values))
        new_probs = jnp.vstack(np.array(new_probs))

        # Calculate the ratios
        ratio = jnp.exp(new_log_probs - old_log_probs)

        # Calculate the surrogate objectives
        clipped_ratio = jnp.clip(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        surrogate1 = ratio * advantages_batch
        surrogate2 = clipped_ratio * advantages_batch

        # Calculate the policy loss
        policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

        # Value loss
        value_loss = jnp.mean((returns_batch - new_values) ** 2)

        # Cross-entropy loss for action probabilities
        entropy = -jnp.sum(new_probs * jnp.log(new_probs + 1e-8), axis=-1)
        entropy_loss = jnp.mean(entropy)

        return policy_loss + self.c1 * value_loss + self.c2 * entropy_loss

    def calculate_advantage(self, trajectory):
        rewards = jnp.array(trajectory.rewards)
        values = jnp.array(trajectory.values)
        next_values = jnp.append(values[1:], 0)
        deltas = rewards + self.gamma * next_values - values
        advantages = jnp.zeros_like(rewards, dtype=jnp.float32)
        adv = 0
        for t in reversed(range(len(rewards))):
            adv = deltas[t] + self.gamma * self.lambda_ * adv
            advantages = advantages.at[t].set(adv)
        returns = advantages + values
        return advantages, returns
