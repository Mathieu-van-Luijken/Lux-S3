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
    key = jax.random.PRNGKey(1480928)

    def __call__(self, params, trajectory, agent):
        observations, actions, old_log_probs, advantages, returns = (
            trajectory.observations,
            trajectory.actions,
            trajectory.log_probs,
            trajectory.advantages,
            trajectory.returns,
        )

        num_samples = len(observations)
        indices = list(range(num_samples))
        random.shuffle(indices)

        for i in range(0, num_samples, self.minibatch_size):
            batch_indices = indices[i : i + self.minibatch_size]
            obs_batch = jnp.stack([observations[j] for j in batch_indices])
            actions_batch = jnp.stack([actions[j] for j in batch_indices])
            old_log_probs_batch = jnp.array([old_log_probs[j] for j in batch_indices])
            advantages_batch = jnp.array([advantages[j] for j in batch_indices])
            returns_batch = jnp.array([returns[j] for j in batch_indices])

            loss, grads = value_and_grad(self.loss_fn)(
                params,
                agent,
                obs_batch,
                actions_batch,
                old_log_probs_batch,
                advantages_batch,
                returns_batch,
            )
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
        return params, opt_state

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
        for i, obs in enumerate(obs_batch):  # Loop over the full trajectory (101 steps)
            action = actions_batch[i]
            (
                unit_positions,
                unit_energies,
                relic_positions,
                tile_board,
                energy_board,
            ) = agent.get_relevant_info(obs)

            for j, unit in enumerate(unit_positions[1]):
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
                    new_log_probs.append(jnp.log(move_probs[0, action[j][0]]))
                    new_values.append(value.item())
                else:
                    new_log_probs.append(0)
                    new_values.append(0)

        # Make the lists into a jnp array for easier computation
        new_log_probs = jnp.vstack(np.array(new_log_probs))
        old_log_probs = jnp.vstack(np.array(old_log_probs_batch))

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
        # TODO this is completely copy pasted does this even work???
        action_log_probs = jnp.take_along_axis(
            new_log_probs, actions_batch[..., None], axis=-1
        ).squeeze(-1)
        cross_entropy_loss = -jnp.mean(action_log_probs)

        return policy_loss + self.c1 * value_loss + self.c2 * cross_entropy_loss

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
