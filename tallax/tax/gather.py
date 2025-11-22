
import functools
import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.utils import NUM_LANES, NUM_SUBLANES, pad_to_tiles


def dense_gather_kernel(values_ref, indices_ref, output_ref):
  """Gather values by indexing in to all of value with a mask.

  This kernel processes multiple tiles of output (NUM_SUBLANES x K).
  It scans across the entire values_ref (which contains full vocab for the corresponding tokens)
  once, updating all output tiles.
  """
  # values_ref: (NUM_SUBLANES, VocabSize)
  # indices_ref: (NUM_SUBLANES, K)
  # output_ref: (NUM_SUBLANES, K)

  num_k_blocks = indices_ref.shape[1] // NUM_LANES

  # Initialize accumulators
  accumulators = [
      jnp.zeros((NUM_SUBLANES, NUM_LANES), dtype=output_ref.dtype)
      for _ in range(num_k_blocks)
  ]

  # Iterate over blocks of values
  for block_offset in range(0, values_ref.shape[1], NUM_LANES):
    block_vals_slice = pl.dslice(block_offset, NUM_LANES)

    # Load values for this block once
    fetched_vals = values_ref[:, block_vals_slice]

    # Apply to all K blocks
    for k_idx in range(num_k_blocks):
        k_offset = k_idx * NUM_LANES
        k_slice = pl.dslice(k_offset, NUM_LANES)

        indices = indices_ref[:, k_slice]

        mask = (indices >= block_offset) & (indices < block_offset + NUM_LANES)

        gathered = jax.vmap(lambda x, y: x[y])(
            fetched_vals,
            indices % NUM_LANES
        )

        accumulators[k_idx] = jnp.where(mask, gathered, accumulators[k_idx])

  # Write out results
  for k_idx in range(num_k_blocks):
      k_offset = k_idx * NUM_LANES
      output_ref[:, pl.dslice(k_offset, NUM_LANES)] = accumulators[k_idx]


@functools.partial(jit, static_argnames=("interpret",))
def gather(
    values,
    indices,
    interpret: bool = False,
):
  """
  Gather values from `values` array using `indices`.

  Args:
      values: Input values [Batch, VocabSize].
      indices: Indices to gather [Batch, K].
      interpret: Run in interpreter mode (CPU compatible).

  Returns:
      Gathered values: [Batch, K].
  """
  num_tokens, vocab_size = values.shape
  num_tokens_idx, k = indices.shape

  if num_tokens != num_tokens_idx:
    raise ValueError(
        f"Batch dimension mismatch: values {num_tokens} vs indices {num_tokens_idx}"
    )

  # Pad dimensions to be multiples of hardware constants
  values = pad_to_tiles(values)
  # indices might need padding only in dim0 and dim1, but dim0 must match values.
  # pad_to_tiles handles both dims.
  indices = pad_to_tiles(indices)

  # If values and indices have different dim0 padding (e.g. one was already padded),
  # pad_to_tiles handles it based on shape.
  # But we need them to have SAME dim0 length if we want to block them together.
  # pad_to_tiles ensures dim0 is multiple of NUM_SUBLANES.
  # If num_tokens was 13, both become 16. Correct.

  pad_tokens = values.shape[0]
  pad_vocab = values.shape[1]
  pad_k = indices.shape[1]

  # Define grid
  # Grid dim 0: Token blocks
  # Grid dim 1: None (we process all K in one go)
  grid = (pad_tokens // NUM_SUBLANES,)

  # Block Specs
  # values: Map (block_idx, ...) -> (NUM_SUBLANES, pad_vocab)
  in_spec_values = pl.BlockSpec(
      (NUM_SUBLANES, pad_vocab),
      lambda i: (i, 0)
  )

  # indices: Map (block_idx, ...) -> (NUM_SUBLANES, pad_k)
  in_spec_indices = pl.BlockSpec(
      (NUM_SUBLANES, pad_k),
      lambda i: (i, 0)
  )

  # output: Same as indices
  out_spec = pl.BlockSpec(
      (NUM_SUBLANES, pad_k),
      lambda i: (i, 0)
  )

  # Call Pallas
  output = pl.pallas_call(
      dense_gather_kernel,
      out_shape=jax.ShapeDtypeStruct(indices.shape, values.dtype),
      in_specs=[in_spec_values, in_spec_indices],
      out_specs=out_spec,
      grid=grid,
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=int(0.9 * 2**27) # ~120MB
      ),
      interpret=interpret
  )(values, indices)

  # Slice to original shape
  return output[:num_tokens, :k]
