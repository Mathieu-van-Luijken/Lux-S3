import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random, grad, value_and_grad
import numpy as np
from typing import Any, Tuple
from utils import sample_action


class PPOLoss(nn.Module):
    epsilon: float = 0.2
    gamma: float = 0.99
    lambda_: float = 0.95
    c1: float = 0.5
    c2: float = 0.01
    key = jax.random.PRNGKey(1480928)

    def __call__(self, params, trajectory, agent):
        observations, actions, old_log_probs, advantages, returns = (
            trajectory.observations,
            trajectory.actions,
            trajectory.log_probs,
            trajectory.advantages,
            trajectory.returns,
        )
        # Recompute log_probs and values for the entire trajectory
        new_values = []
        new_log_probs = []
        for i in range(len(observations)):  # Loop over the full trajectory (101 steps)
            obs = observations[i]
            action = actions[i]
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

        new_values = jnp.vstack(jnp.array(new_values))
        new_log_probs = jnp.vstack(jnp.array(new_log_probs))

        # PPO Clipping
        ratio = jnp.exp(new_log_probs - old_log_probs)
        clip_adv = jnp.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -jnp.mean(jnp.minimum(ratio * advantages, clip_adv))

        # Critic Loss (MSE loss between predicted values and returns)
        critic_loss = jnp.mean((new_values - returns) ** 2)

        # Entropy Loss (encourages exploration)
        entropy_loss = -jnp.mean(
            jnp.sum(jnp.exp(new_log_probs) * new_log_probs, axis=-1)
        )

        # Combined loss
        total_loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss
        return total_loss

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
