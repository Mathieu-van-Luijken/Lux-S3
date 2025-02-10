import jax
import jax.numpy as jnp
import jax.random as random


def sample_action(key, move_probs, num_units):
    key, subkey = jax.random.split(key)

    existing_units = move_probs.shape[0]
    final_moves = jnp.zeros((num_units, 3), dtype=jnp.int16)
    sampled_moves = random.categorical(key, logits=jnp.log(move_probs))
    final_moves = final_moves.at[:existing_units, 0].set(sampled_moves)
    final_moves = final_moves.at[final_moves[:, 0] == 6, 0].set(0)
    return final_moves
