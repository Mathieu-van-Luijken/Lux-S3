import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random, grad, value_and_grad
import random
import numpy as np
from typing import Any, Tuple
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
        observations, actions, old_log_probs, advantages, returns = (
            trajectory.observations,
            trajectory.actions,
            trajectory.log_probs,
            trajectory.advantages,
            trajectory.returns,
        )

        num_samples = len(observations)
        perm = jax.random.permutation(key, num_samples)

        shuffled_obs = [observations[i] for i in perm]

        shuffled_actions = jnp.array([actions[i] for i in perm])
        shuffled_old_log_probs = jnp.array([old_log_probs[i] for i in perm])
        shuffled_advantages = jnp.array([advantages[i] for i in perm])
        shuffled_returns = jnp.array([returns[i] for i in perm])

        def batch_update(carry, batch_idx):
            params, opt_state = carry

            current_batch_size = jnp.minimum(
                self.minibatch_size, num_samples - batch_idx
            )
            obs_batch = shuffled_obs[batch_idx : batch_idx + current_batch_size]
            actions_batch = jax.lax.dynamic_slice(
                shuffled_actions, (batch_idx,), (self.minibatch_size,)
            )
            old_log_probs_batch = jax.lax.dynamic_slice(
                shuffled_old_log_probs, (batch_idx,), (self.minibatch_size,)
            )
            advantages_batch = jax.lax.dynamic_slice(
                shuffled_advantages, (batch_idx,), (self.minibatch_size,)
            )
            returns_batch = jax.lax.dynamic_slice(
                shuffled_returns, (batch_idx,), (self.minibatch_size,)
            )

            loss, grads = value_and_grad(self.loss_fn)(
                params,
                agent,
                obs_batch,
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
        obs_batch,
        actions_batch,
        old_log_probs_batch,
        advantages_batch,
        returns_batch,
    ):
        # Recompute log_probs and values for the entire trajectory
        new_values = []
        new_log_probs = []
        new_probs = []
        for i, obs in enumerate(obs_batch):
            action = actions_batch[i]

            for j, unit in enumerate(unit_positions[1]):
                new_value = 0
                if jnp.any(unit[1] >= 0):
                    unit_info = jnp.append(unit, unit_energies[1][j])[None, :]
                    value, move_probs = agent.apply(
                        params,
                        unit_positions,
                        unit_energies,
                        relic_positions,
                        tile_board,
                        energy_board,
                        unit_info,
                    )
                    new_probs.append(
                        move_probs.primal
                    )  # NOTE i have absolutely no idea why it is primal
                    new_log_probs.append(jnp.log(move_probs[0, action[j][0]]).item())
                    new_value = value.item()
                else:
                    new_log_probs.append(0)

            new_values.append(new_value)

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
