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
