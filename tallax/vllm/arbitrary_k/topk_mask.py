"""Top-k masking via binary search over the full vocabulary.

Finds the k'th largest threshold value using binary search in f32 space,
then applies stable masking using parallel chunk-based boundary detection.
Never sorts the input.
"""

import functools
import math
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.vllm.utils.binary_search import binary_search
from tallax.tax.utils import (
  NUM_LANES,
  get_dtype_info,
  map_reduce,
  to_32bit_dtype,
)


def _find_boundary_chunk(
  ref,
  map_fn,
  target,
  chunk_size: int,
  active_chunk: jax.Array | None = None,
  ref_offset: jax.Array | int = 0,
):
  """Find which chunk contains the target boundary according to map_fn.

  Splits vocabulary into chunks, counts matches in each chunk,
  builds cumulative sums, and iterates to find which chunk contains
  the target count.

  Args:
    ref: Reference array of shape [batch, vocab_size]
    map_fn: Unary function mapping chunks to binary counters
    target: Target count (shape [batch, 1])
    chunk_size: Size of each chunk
    active_chunk: Optional subset of ref to search in
    ref_offset: Offset into ref for indexing

  Returns:
    Tuple of (ref_offset, boundary_slice, target)
  """
  arr = ref if active_chunk is None else active_chunk
  num_chunks = arr.shape[1] // chunk_size
  chunks = [
    arr[:, i * chunk_size : (i + 1) * chunk_size].astype(
      to_32bit_dtype(arr.dtype)
    )
    for i in range(num_chunks)
  ]
  assert chunk_size % NUM_LANES == 0

  num_matches = [map_fn(chunk).sum(axis=1, keepdims=True) for chunk in chunks]

  cumsums = [num_matches[0]]
  for i in range(1, len(num_matches)):
    cumsums.append(cumsums[i - 1] + num_matches[i])

  boundary_idx = sum((c < target) for c in cumsums)
  target -= sum((i == (boundary_idx - 1)) * c for i, c in enumerate(cumsums))

  batch_size = ref.shape[0]
  iota0, iota1 = (
    jax.lax.broadcasted_iota(jnp.int32, (batch_size, chunk_size), dim)
    for dim in (0, 1)
  )

  ref_offset += boundary_idx * chunk_size
  boundary_slices = [
    ref[
      :, pl.dslice(pl.multiple_of(ref_offset[i, 0], chunk_size), chunk_size)
    ].astype(to_32bit_dtype(ref.dtype))
    for i in range(batch_size)
  ]
  boundary_slice = boundary_slices[0]
  for i in range(1, batch_size):
    boundary_slice = jnp.where(iota0 == i, boundary_slices[i], boundary_slice)
  if num_chunks * chunk_size != arr.shape[1]:
    boundary_slice = jnp.where(
      (ref_offset[:, :1] + iota1) < ref.shape[1],
      boundary_slice,
      get_dtype_info(boundary_slice).min,
    )
  return ref_offset, boundary_slice, target


def find_boundary_idx(ref_or_arr, map_fn, target):
  """Find the lowest idx where map_fn(ref[...]).cumsum(1) >= target.

  Uses a two-level chunk search: first finds the right chunk of chunks,
  then the right chunk, then within-tile cumsum for the exact index.

  Args:
    ref_or_arr: Pallas ref or array of shape [batch, vocab_size]
    map_fn: Maps chunks to binary counts
    target: Target cumulative sum value

  Returns:
    Index array of shape [batch, NUM_LANES]
  """
  if "Ref" not in str(ref_or_arr):
    arr = ref_or_arr

    def scoped_body(scoped_ref):
      scoped_ref[...] = arr
      return find_boundary_idx(scoped_ref, map_fn, target)

    return pl.run_scoped(scoped_body, pltpu.VMEM(arr.shape, arr.dtype))
  ref = ref_or_arr

  assert ref.ndim == 2
  ref_offset, boundary_slice, target = _find_boundary_chunk(
    ref,
    map_fn=map_fn,
    target=target,
    chunk_size=int(math.sqrt(ref.shape[1] // NUM_LANES)) * NUM_LANES,
  )
  ref_offset, boundary_slice, target = _find_boundary_chunk(
    ref,
    map_fn=map_fn,
    target=target,
    chunk_size=NUM_LANES,
    ref_offset=ref_offset,
    active_chunk=boundary_slice,
  )
  # Within tile cumsum check
  iota1 = jax.lax.broadcasted_iota(jnp.int32, (ref.shape[0], NUM_LANES), 1)
  mapped_slice = map_fn(boundary_slice)
  cumsums = [
    (mapped_slice * (iota1 <= i)).sum(1, keepdims=True)
    for i in range(NUM_LANES)
  ]
  return ref_offset + sum((c < target) for c in cumsums)


def topk_mask(
  logits_ref,
  k_ref,
  *,
  replace_val: float,
  stable: bool,
  underlying_dtype=None,
):
  """Apply top-k mask using binary search to find the k'th largest threshold.

  Args:
    logits_ref: Input logits reference [batch, vocab_size]
    k_ref: Number of top elements to keep [batch, 1]
    replace_val: Replacement value for masked elements
    stable: Whether to use stable masking (exactly k elements kept)
    underlying_dtype: Original dtype if logits were cast (for bf16 search convergence)

  Returns:
    Masked logits with only top-k values preserved
  """
  logits = logits_ref[...].astype(jnp.float32)
  k = k_ref[...]
  bound_shape = (logits.shape[0], NUM_LANES)
  k = jnp.broadcast_to(k, bound_shape)
  predicate_fn = (
    lambda pivot: map_reduce(
      logits,
      lambda chunk: (chunk > pivot).astype(jnp.int32),
      reduce_fn="sum",
    )
    < k
  )
  finfo = jnp.finfo(logits_ref.dtype)
  _, threshold, _ = binary_search(
    predicate_fn,
    *(jnp.full(bound_shape, v, logits.dtype) for v in (finfo.min, finfo.max)),
    num_iter=(underlying_dtype.itemsize * 8)
    if underlying_dtype is not None
    else 32,
    underlying_dtype=underlying_dtype,
  )

  assert logits.shape[1] % NUM_LANES == 0
  if not stable:
    mask = logits >= pltpu.repeat(
      threshold,
      logits.shape[1] // NUM_LANES,
      axis=1,
    )
  else:
    boundary_idx = find_boundary_idx(
      logits_ref,
      map_fn=lambda chunk: (
        chunk == pltpu.repeat(threshold, chunk.shape[1] // NUM_LANES, 1)
      ).astype(jnp.int32),
      target=k
      - map_reduce(
        logits,
        lambda chunk: (chunk > threshold).astype(jnp.int32),
        reduce_fn="sum",
      ),
    )
    threshold = pltpu.repeat(
      threshold,
      logits.shape[1] // NUM_LANES,
      axis=1,
    )
    boundary_idx = pltpu.repeat(
      boundary_idx, logits.shape[1] // NUM_LANES, axis=1
    )
    mask = (logits > threshold) | (
      (logits == threshold)
      & (
        jax.lax.broadcasted_iota(jnp.int32, logits_ref.shape, 1) <= boundary_idx
      )
    )
  return jnp.where(mask, logits, replace_val).astype(logits_ref.dtype)


def topk_mask_pallas_kernel(
  logits_ref,
  k_ref,
  output_ref,
  *,
  replace_val: float,
  stable: bool,
):
  output_ref[...] = topk_mask(
    logits_ref, k_ref, replace_val=replace_val, stable=stable
  )


@functools.partial(
  jax.jit, static_argnames=["replace_val", "stable", "interpret"]
)
def topk_mask_pallas(
  x: jax.Array,
  k: int,
  replace_val: float = -1e12,
  stable: bool = True,
  interpret: bool = False,
) -> jax.Array:
  """Pallas-based topk mask with parallel chunk-based reduction.

  Args:
    x: Input array of shape [batch, vocab_size]
    k: Number of top elements
    replace_val: Value for masked elements
    stable: Whether to use stable masking
    interpret: Whether to use interpret mode

  Returns:
    Masked array
  """
  batch_size, _vocab_size = x.shape
  k = jnp.broadcast_to(k, (batch_size, 1))
  output_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
  return pl.pallas_call(
    functools.partial(
      topk_mask_pallas_kernel,
      replace_val=replace_val,
      stable=stable,
    ),
    compiler_params=pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 2**27)),
    out_shape=output_shape,
    interpret=interpret,
  )(x, k)
