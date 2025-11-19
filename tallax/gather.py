
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .utils import is_cpu_platform


def _dense_gather_kernel(values_ref, indices_ref, output_ref, *, block_size: int):
  """Gather values by indexing in to all of value with a mask, rather than a single gather per index."""
  for token_offset in range(0, values_ref.shape[0], block_size):
    token_slice = pl.dslice(token_offset, block_size)
    output = jnp.zeros((block_size, output_ref.shape[1]), values_ref.dtype)
    indices = indices_ref[token_slice]

    for block_offset in range(0, values_ref.shape[1], block_size):
      mask = (indices >= block_offset) & (indices < block_offset + block_size)
      output = jnp.where(
          mask,
          jax.vmap(lambda x, y: x[y])(
              values_ref[
                  token_slice,
                  block_offset: block_offset + block_size
              ],
              indices % block_size
          ),
          output,
      )

    output_ref[token_slice] = output


def gather_pallas(values, indices, *, vmem_limit_bytes: int | None = None):
  """High-level interface for dense gather on TPU."""
  block_size = values.shape[0]

  if vmem_limit_bytes is not None:
    # Heuristic for block_size based on VMEM limit.
    # The kernel loads a block of `values` of size `(block_size, block_size)`.
    block_size_limit = int((vmem_limit_bytes / values.dtype.itemsize) ** 0.5)
    block_size = min(
        block_size,
        max(1, block_size_limit)
    )

  output_shape = jax.ShapeDtypeStruct(indices.shape, values.dtype)
  kernel = pl.pallas_call(
      functools.partial(_dense_gather_kernel, block_size=block_size),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.VMEM),
          pl.BlockSpec(memory_space=pltpu.VMEM),
      ],
      out_shape=output_shape,
      out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
      grid=(1,),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes
      ),
      interpret=is_cpu_platform(),
  )
  return kernel(values, indices)
