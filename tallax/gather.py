
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .utils import is_cpu_platform, NUM_LANES, NUM_SUBLANES


def dense_gather_kernel(values_ref, indices_ref, output_ref):
  """Gather values by indexing in to all of value with a mask, rather than a single gather per index."""
  for token_offset in range(0, values_ref.shape[0], NUM_SUBLANES):
    token_slice = pl.dslice(token_offset, NUM_SUBLANES)
    indices = indices_ref[token_offset: token_offset + NUM_SUBLANES]
    output = jnp.zeros_like(indices, dtype=values_ref.dtype)

    for block_offset in range(0, values_ref.shape[1], NUM_LANES):
      mask = (indices >= block_offset) & (indices < block_offset + NUM_LANES)
      gathered_values = jax.vmap(lambda x, y: x[y])(
          values_ref[
              token_offset: token_offset + NUM_SUBLANES,
              block_offset: block_offset + NUM_LANES
          ],
          indices % NUM_LANES
      )
      output = jnp.where(mask, gathered_values, output)

    output_ref[token_slice] = output.astype(output_ref.dtype)


def gather_pallas(values, indices):
  """High-level interface for dense gather on TPU."""
  output_shape = jax.ShapeDtypeStruct(indices.shape, values.dtype)
  return pl.pallas_call(
      dense_gather_kernel,
      in_specs=[
          pl.BlockSpec(values.shape, lambda i: (0, 0)),
          pl.BlockSpec(indices.shape, lambda i: (0, 0)),
      ],
      out_shape=output_shape,
      grid=(1,),
      compiler_params=pltpu.CompilerParams(),
      interpret=is_cpu_platform(),
  )(values, indices)
