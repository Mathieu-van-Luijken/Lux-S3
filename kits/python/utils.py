import jax
import jax.numpy as jnp


def sample_action(key, move_probs):
    key, subkey = jax.random.split(key)
    move_action = jax.random.choice(
        subkey, move_probs.shape[-1], p=move_probs.flatten()
    )

    if move_action == 5:
        key, subkey_x, subkey_y = jax.random.split(key, 3)
        move_action = 0

    return jnp.array([move_action, 0, 0])
