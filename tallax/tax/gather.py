
import functools
import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.utils import NUM_LANES, NUM_SUBLANES


def dense_gather_kernel(values_ref, indices_ref, output_ref):
  """Gather values by indexing in to all of value with a mask.

  This kernel processes a single tile of output (NUM_SUBLANES x NUM_LANES).
  It scans across the entire values_ref (which contains full vocab for the corresponding tokens).
  """
  # values_ref: (NUM_SUBLANES, VocabSize) - VocabSize must be multiple of NUM_LANES
  # indices_ref: (NUM_SUBLANES, NUM_LANES)
  # output_ref: (NUM_SUBLANES, NUM_LANES)

  output = jnp.zeros_like(output_ref)
  indices = indices_ref[...]

  # Iterate over blocks of values
  for block_offset in range(0, values_ref.shape[1], NUM_LANES):
    block_vals_slice = pl.dslice(block_offset, NUM_LANES)

    # Check if indices fall within this block
    mask = (indices >= block_offset) & (indices < block_offset + NUM_LANES)

    # Load values for this block
    # Note: values_ref corresponds to the rows for this token block
    fetched_vals = values_ref[:, block_vals_slice]

    # Gather from the loaded block using local indices
    gathered = jax.vmap(lambda x, y: x[y])(
        fetched_vals,
        indices % NUM_LANES
    )

    output = jnp.where(mask, gathered, output)

  output_ref[...] = output


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

  # Pad tokens to NUM_SUBLANES
  pad_tokens = (num_tokens + NUM_SUBLANES - 1) // NUM_SUBLANES * NUM_SUBLANES
  if pad_tokens != num_tokens:
    values = jnp.pad(values, ((0, pad_tokens - num_tokens), (0, 0)), constant_values=0)
    indices = jnp.pad(indices, ((0, pad_tokens - num_tokens), (0, 0)), constant_values=0)

  # Pad vocab to NUM_LANES
  pad_vocab = (vocab_size + NUM_LANES - 1) // NUM_LANES * NUM_LANES
  if pad_vocab != vocab_size:
    values = jnp.pad(values, ((0, 0), (0, pad_vocab - vocab_size)), constant_values=0)

  # Pad K to NUM_LANES
  pad_k = (k + NUM_LANES - 1) // NUM_LANES * NUM_LANES
  if pad_k != k:
    indices = jnp.pad(indices, ((0, 0), (0, pad_k - k)), constant_values=0)

  # Define grid
  # Grid dim 0: Token blocks
  # Grid dim 1: Output K blocks
  grid = (pad_tokens // NUM_SUBLANES, pad_k // NUM_LANES)

  # Block Specs
  # values: Map (block_idx_0, any) -> (NUM_SUBLANES, pad_vocab)
  # We want the token block corresponding to dim0 of grid, and ALL of dim1 (Vocab)
  in_spec_values = pl.BlockSpec(
      (NUM_SUBLANES, pad_vocab),
      lambda i, j: (i, 0)
  )

  # indices: Map (block_idx_0, block_idx_1) -> (NUM_SUBLANES, NUM_LANES)
  in_spec_indices = pl.BlockSpec(
      (NUM_SUBLANES, NUM_LANES),
      lambda i, j: (i, j)
  )

  # output: Same as indices
  out_spec = pl.BlockSpec(
      (NUM_SUBLANES, NUM_LANES),
      lambda i, j: (i, j)
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
