import jax
import flax.linen as nn
import jax.numpy as jnp
from jax import random, grad, value_and_grad
from flax.training import train_state
import optax
import numpy as np
from typing import Any, Tuple


class PPOAgent(nn.Module):

    @nn.compact
    def cnn_embedder(self, board_state_tensor, unit_info):
        """Embeds the board state tensor into a 128-dimensional space."""
        x = board_state_tensor.transpose(0, 2, 3, 1)  # Add channel dimension
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=128)(x)
        board_embedding = nn.relu(x)
        embedding = jnp.concat([board_embedding, unit_info], axis=-1)
        return embedding

    @nn.compact
    def critic(self, embedding):
        """Computes the value estimation."""
        x = nn.Dense(features=64)(embedding)
        x = nn.relu(x)
        value = nn.Dense(features=1)(x)
        return value

    @nn.compact
    def actor(self, embedding):
        """Computes the action logits."""
        x = nn.Dense(features=64)(embedding)
        x = nn.relu(x)
        action_logits = nn.Dense(features=6)(x)
        x_coord = nn.Dense(features=4)(x)
        y_coord = nn.Dense(features=4)(x)
        return action_logits, x_coord, y_coord

    @nn.compact
    def __call__(self, board_state_tensor, unit_info):
        """Forward pass of the PPO agent."""
        embedding = self.cnn_embedder(board_state_tensor, unit_info=unit_info)
        value = self.critic(embedding)
        action_logits, x_coord_logits, y_coord_logits = self.actor(embedding)
        return value, action_logits, x_coord_logits, y_coord_logits
