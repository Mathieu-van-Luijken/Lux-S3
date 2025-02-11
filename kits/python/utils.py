import jax
import jax.numpy as jnp
import jax.random as random


def sample_action(key, move_probs, num_units):
    key, subkey = jax.random.split(key)

    existing_units = move_probs.shape[0]
    final_moves = jnp.zeros((num_units, 3), dtype=jnp.int32)
    sampled_moves = random.categorical(key, logits=jnp.log(move_probs))
    final_moves = final_moves.at[:existing_units, 0].set(sampled_moves)
    final_moves = final_moves.at[final_moves[:, 0] == 6, 0].set(0)
    return final_moves


def calc_log_probs(move_probs, actions, num_units):

    sampled_actions = actions[:, 0]
    existing_units = move_probs.shape[0]
    output_log_probs = jnp.zeros(num_units, dtype=jnp.float32)

    if existing_units > 0:
        log_probs = jnp.log(move_probs + 1e-8)
        selected_log_probs = log_probs[jnp.arange(num_units), sampled_actions][
            :existing_units
        ]
        output_log_probs = output_log_probs.at[:existing_units].set(selected_log_probs)
    return output_log_probs
