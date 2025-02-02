import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random, grad, value_and_grad
import numpy as np
from typing import Any, Tuple
from ...src.luxai_s3.env import LuxAIS3Env


class PPOLoss(nn.Module):
    epsilon: float = 0.2
    c1: float = 0.5
    c2: float = 0.01

    def __call__(
        self,
        params,
        observations,
        actions,
        log_probs_old,
        advantages,
        returns,
        model_apply_fn,
    ):

        logits, values = model_apply_fn(params, observations)

        log_probs = jnp.sum(
            jnp.take_along_axis(
                jax.nn.log_softmax(logits), actions[..., None], axis=-1
            ).squeeze(-1),
            axis=-1,
        )
        policy_loss = self.policy_loss(
            log_probs=log_probs, log_probs_old=log_probs_old, advantages=advantages
        )
        critic_loss = self.critic_loss(values=values, returns=returns)
        entropy_loss = self.entropy_loss(logits=logits)

        return policy_loss + critic_loss + entropy_loss

    def policy_loss(self, log_probs, log_probs_old, advantages):
        ratio = jnp.exp(log_probs - log_probs_old)
        clipped_ratio = jnp.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = -jnp.mean(jnp.min(ratio * advantages, clipped_ratio * advantages))
        return policy_loss

    def critic_loss(self, values, returns):
        value_loss = self.c1 * jnp.mean((returns - values) ** 2)
        return value_loss

    def entropy_loss(self, logits):
        log_probs = jax.nn.log_softmax(logits)
        probs = jnp.exp(log_probs)
        entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))
        entropy_bonus = -self.c2 * entropy
        return entropy_bonus

    def compute_advantages(
        self, rewards, values, next_values, dones, gae_gamma=0.99, gae_lambda=0.95
    ):
        deltas = rewards + gae_gamma * next_values * (1 - dones) - values
        advantages = jax.lax.scan(
            lambda acc, delta: gae_gamma * gae_lambda * acc * (1 - dones) + delta,
            jnp.zeros_like(deltas[-1]),
            deltas[::-1],
        )[0][::-1]
        returns = advantages + values
        return advantages, returns

    def compute_log_probs(self, action_probs, actions):
        """Compute log probabilities for sampled actions."""
        move_probs = action_probs["move_probs"]
        positional_probs_x = action_probs["positional_probs_x"]
        positional_probs_y = action_probs["positional_probs_y"]

        move_action = actions[0]
        log_prob_move = jnp.log(move_probs[move_action])

        log_prob_positional_x = 0
        log_prob_positional_y = 0
        if move_action == 5:
            positional_action_x = actions[1]
            positional_action_y = actions[2]
            log_prob_positional_x = jnp.log(positional_probs_x[positional_action_x])
            log_prob_positional_y = jnp.log(positional_probs_y[positional_action_y])

        return log_prob_move + log_prob_positional_x + log_prob_positional_y
