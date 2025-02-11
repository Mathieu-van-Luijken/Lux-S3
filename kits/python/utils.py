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


def dynamic_slice(
    unit_positions,
    player_unit_positions,
    unit_energies,
    relic_positions,
    tile_board,
    energy_board,
    actions,
    old_log_probs,
    advantages,
    returns,
    minibatch_size,
    batch_idx,
):
    # Use dynamic slice to select the batch for each array
    unit_positions_batch = jax.lax.dynamic_slice(
        unit_positions,
        start_indices=(batch_idx, 0, 0, 0),
        slice_sizes=(minibatch_size, 2, 16, 2),
    )

    player_unit_positions_batch = jax.lax.dynamic_slice(
        player_unit_positions,
        start_indices=(batch_idx, 0, 0),
        slice_sizes=(minibatch_size, 16, 2),
    )

    unit_energies_batch = jax.lax.dynamic_slice(
        unit_energies,
        start_indices=(batch_idx, 0, 0),
        slice_sizes=(minibatch_size, 2, 16),
    )

    relic_positions_batch = jax.lax.dynamic_slice(
        relic_positions,
        start_indices=(batch_idx, 0, 0),
        slice_sizes=(minibatch_size, 6, 2),
    )

    tile_board_batch = jax.lax.dynamic_slice(
        tile_board,
        start_indices=(batch_idx, 0, 0),
        slice_sizes=(minibatch_size, 24, 24),
    )

    energy_board_batch = jax.lax.dynamic_slice(
        energy_board,
        start_indices=(batch_idx, 0, 0),
        slice_sizes=(minibatch_size, 24, 24),
    )

    actions_batch = jax.lax.dynamic_slice(
        actions,
        start_indices=(batch_idx, 0, 0),
        slice_sizes=(minibatch_size, 16, 3),
    )

    old_log_probs_batch = jax.lax.dynamic_slice(
        old_log_probs,
        start_indices=(batch_idx, 0),
        slice_sizes=(minibatch_size, 16),
    )

    advantages_batch = jax.lax.dynamic_slice(
        advantages,
        start_indices=(batch_idx,),
        slice_sizes=(minibatch_size,),
    )

    returns_batch = jax.lax.dynamic_slice(
        returns,
        start_indices=(batch_idx,),
        slice_sizes=(minibatch_size,),
    )
    return (
        unit_positions_batch,
        player_unit_positions_batch,
        unit_energies_batch,
        relic_positions_batch,
        tile_board_batch,
        energy_board_batch,
        actions_batch,
        old_log_probs_batch,
        advantages_batch,
        returns_batch,
    )
