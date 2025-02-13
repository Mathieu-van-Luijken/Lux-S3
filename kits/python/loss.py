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

    def __init__(self):
        self.optimizer = optax.adam(self.learning_rate)
        self.key = jax.random.PRNGKey(1480928)

    def __call__(self, params, trajectory, agent, key, opt_state):
        (
            player_unit_positions,
            board_state,
            num_active_units,
            actions,
            log_probs,
            advantages,
            returns,
        ) = (
            trajectory.player_unit_positions,
            trajectory.board_state,
            trajectory.num_active_units,
            trajectory.actions,
            trajectory.log_probs,
            trajectory.advantages,
            trajectory.returns,
        )

        num_samples = player_unit_positions.shape[0]
        perm = jax.random.permutation(key, num_samples)

        # Shuffle the data to obtain random batches
        shuffled_player_unit_positions = player_unit_positions[perm]
        shuffled_board_state = board_state[perm]
        shuffled_num_active_units = num_active_units[perm]
        shuffled_actions = actions[perm]
        shuffled_log_probs = log_probs[perm]
        shuffled_advantages = advantages[perm]
        shuffled_returns = returns[perm]

        def batch_update(carry, batch_idx):
            params, opt_state = carry
            (
                player_unit_positions_batch,
                board_state_batch,
                num_active_units_batch,
                actions_batch,
                old_log_probs_batch,
                advantages_batch,
                returns_batch,
            ) = dynamic_slice(
                player_unit_positions=shuffled_player_unit_positions,
                board_state=shuffled_board_state,
                num_active_units=shuffled_num_active_units,
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
                player_unit_positions_batch,
                board_state_batch,
                num_active_units_batch,
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
        player_unit_positions_batch,
        board_state_batch,
        num_active_units_batch,
        actions_batch,
        old_log_probs_batch,
        advantages_batch,
        returns_batch,
    ):

        def agent_call(carry, batch_idx):
            (
                player_unit_positions,
                board_state,
                _,
                _,
                _,
                _,
                _,
            ) = dynamic_slice(
                player_unit_positions=player_unit_positions_batch,
                board_state=board_state_batch,
                num_active_units=num_active_units_batch,
                actions=actions_batch,
                old_log_probs=old_log_probs_batch,
                advantages=advantages_batch,
                returns=returns_batch,
                minibatch_size=1,
                batch_idx=batch_idx,
            )

            new_value, new_move_probs = agent.apply(
                params,
                board_state=board_state.squeeze(0),
                player_unit_positions=player_unit_positions.squeeze(),
            )
            return carry, (new_value, new_move_probs)

        # Initialize carry as a tuple of zeros with the same shape as value and move_probs
        initial_carry = None

        # Run the scan over the batch
        carry, (values, move_probs) = jax.lax.scan(
            agent_call, initial_carry, jnp.arange(player_unit_positions_batch.shape[0])
        )

        # Get all new log probs
        action_indices = actions_batch[:, :, 0]
        chosen_move_probs = jnp.take_along_axis(
            move_probs, action_indices[..., None], axis=-1
        )
        chosen_move_probs = chosen_move_probs[..., 0]
        new_log_probs = jnp.log(chosen_move_probs)

        # Compute the actual values
        new_values = values.squeeze()

        # Not all units actually existed at prediction time so mask them out
        active_units_mask = (
            jnp.arange(old_log_probs_batch.shape[1])
            < num_active_units_batch.squeeze()[:, None]
        )
        ratio = jnp.exp(new_log_probs - old_log_probs_batch) * active_units_mask

        # Normalize the values such that only relevant units are used in the calculation
        normalized_ratio = normalized_ratio = (
            jnp.sum(ratio, axis=1) / num_active_units_batch.squeeze()
        )

        # Calculate the surrogate objectives
        clipped_ratio = jnp.clip(
            normalized_ratio, 1.0 - self.epsilon, 1.0 + self.epsilon
        )
        surrogate1 = normalized_ratio * advantages_batch
        surrogate2 = clipped_ratio * advantages_batch

        # Calculate the policy loss
        policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

        # Value loss
        value_loss = jnp.mean((returns_batch - new_values) ** 2)

        # Cross-entropy loss for action probabilities
        entropy = -jnp.sum(move_probs * jnp.log(move_probs + 1e-8), axis=-1)
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

        # Normalize advantages: subtract mean and divide by std (across the entire batch)
        advantages = (advantages - jnp.mean(advantages)) / (
            jnp.std(advantages) + 1e-8
        )  # Avoid division by zero
        returns = advantages + values
        return advantages, returns
