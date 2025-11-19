
import functools
import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .sort import bitonic_sort
from .utils import _unrolled_fori_loop, NUM_LANES, NUM_SUBLANES


def blockwise_topk(
    logits,
    k: int,
    block_topk_values=None,
    block_topk_indices=None,
    start_k: int = 0,
    num_blocks: int = NUM_LANES,
    mode: str = "jax",
):
  """
  Compute blockwise top-k using a sinking sort approach.

  Args:
      logits: Input logits to find top-k from
      k: Number of top elements to find
      block_topk_values: Pre-allocated buffers for values
      block_topk_indices: Pre-allocated buffers for indices
      start_k: Starting position (for incremental top-k)
      num_blocks: Number of blocks to process
      mode: "jax" or "pallas" execution mode
  """
  num_tokens = logits.shape[0]

  if start_k != 0 and (block_topk_values is None or block_topk_indices is None):
    raise ValueError(
        "start_k > 0 requires pre-computed buffers in "
        "block_topk_values and block_topk_indices"
    )

  if mode == "jax":
    block_topk_values = [
        jnp.full(
            (num_tokens, num_blocks),
            jnp.finfo(logits.dtype).min,
            dtype=logits.dtype
        )
        for _ in range(k)
    ]
    block_topk_indices = [
        jnp.full((num_tokens, num_blocks), 0, dtype=jnp.int32)
        for _ in range(k)
    ]
  elif mode == "pallas":
    if block_topk_values is None or block_topk_indices is None:
      raise ValueError(
          "Pallas mode requires pre-allocated buffers"
      )

  def process_block(block_idx, carry):
    """Process a single tile with sinking sort."""
    values_list, indices_list = carry

    # Extract current block
    if mode == "pallas":
      current_values = logits[..., pl.dslice(num_blocks * block_idx, num_blocks)]
    elif mode == "jax":
      current_values = jax.lax.dynamic_slice_in_dim(
          logits, block_idx * num_blocks, num_blocks, axis=1
      )
    else:
      raise ValueError("mode must be 'pallas' or 'jax'")

    current_indices = jnp.full((num_tokens, num_blocks), block_idx, jnp.int32)

    # Sinking sort: compare and swap through k levels
    for level in range(k):
      if level < start_k:
        # Invalidate already-found elements
        current_values = jnp.where(
            current_indices == indices_list[level],
            float("-inf"),
            current_values
        )
      else:
        # Exchange with stored top-k
        mask = current_values > values_list[level]

        values_list[level], current_values = (
            jnp.where(m, current_values, values_list[level])
            for m in (mask, ~mask)
        )
        indices_list[level], current_indices = (
            jnp.where(m, current_indices, indices_list[level])
            for m in (mask, ~mask)
        )

    return (values_list, indices_list)

  return _unrolled_fori_loop(
      logits.shape[-1] // num_blocks,
      process_block,
      (block_topk_values, block_topk_indices),
      unroll=16,
  )


def dense_gather_kernel(values_ref, indices_ref, output_ref):
  """Gather values by indexing in to all of value with a mask, rather than a single gather per index."""
  for token_offset in range(0, values_ref.shape[0], NUM_SUBLANES):
    token_slice = pl.dslice(token_offset, NUM_SUBLANES)
    output = jnp.zeros((NUM_SUBLANES, NUM_LANES), values_ref.dtype)
    indices = indices_ref[token_offset: token_offset + NUM_SUBLANES]

    for block_offset in range(0, values_ref.shape[1], NUM_LANES):
      mask = (indices >= block_offset) & (indices < block_offset + NUM_LANES)
      output = jnp.where(
          mask,
          jax.vmap(lambda x, y: x[y])(
              values_ref[
                  token_offset: token_offset + NUM_SUBLANES,
                  block_offset: block_offset + NUM_LANES
              ],
              indices % NUM_LANES
          ),
          output,
      )

    output_ref[token_slice] = output[:, :output_ref.shape[1]].astype(output_ref.dtype)


def topk_blockwise_superset_kernel(
    logits_ref,
    topk_values_ref,
    topk_indices_ref,
    max_depth_ref,
    block_topm_values_ref,
    block_topm_indices_ref,
    termination_flag_ref,
    *,
    k: int = 64,
    block_topk_schedule: tuple[int] | None = None,
    topk_schedule: tuple[int] | None = None,
):
  """
  Compute blockwise top-k supersets until global top-k is guaranteed.

  This uses an adaptive algorithm that incrementally increases m until
  the blockwise top-m's provably contain the global top-k.
  """
  # Initialize buffers
  block_size = logits_ref.shape[0]
  shape = (block_size, block_topm_values_ref.shape[1])

  token_slice = pl.dslice(pl.program_id(0) * block_size, block_size)

  block_topm_values_ref[token_slice] = jnp.full(
      shape, jnp.finfo(jnp.float32).min, dtype=jnp.float32
  )
  block_topm_indices_ref[token_slice] = jnp.full(shape, 0, dtype=jnp.int32)

  for i in range(block_size):
    max_depth_ref[pl.program_id(0) * block_size + i] = k

  termination_flag_ref[0] = 0

  # Schedule of progressively larger m values
  if block_topk_schedule is None:
    block_topk_schedule = (5, 7, 9, 12)
  block_topk_schedule = (0,) + block_topk_schedule + (k,)

  # Incremental blockwise top-k computation
  for completed_m, target_m in zip(block_topk_schedule, block_topk_schedule[1:]):

    @pl.when(termination_flag_ref[0] == 0)
    def _():
      # Compute blockwise top-m
      topk_vals, topk_idxs = blockwise_topk(
          logits_ref,
          block_topk_values=[
              block_topm_values_ref[
                  token_slice, pl.dslice(i * NUM_LANES, NUM_LANES)
              ].astype(jnp.float32)
              for i in range(target_m)
          ],
          block_topk_indices=[
              block_topm_indices_ref[
                  token_slice, pl.dslice(i * NUM_LANES, NUM_LANES)
              ]
              for i in range(target_m)
          ],
          k=target_m,
          num_blocks=NUM_LANES,
          start_k=completed_m,
          mode="pallas",
      )

      # Store results
      for i in range(completed_m, target_m):
        block_topm_values_ref[
            token_slice, pl.dslice(i * NUM_LANES, NUM_LANES)
        ] = topk_vals[i].astype(block_topm_values_ref.dtype)
        block_topm_indices_ref[
            token_slice, pl.dslice(i * NUM_LANES, NUM_LANES)
        ] = topk_idxs[i].astype(block_topm_indices_ref.dtype)

      # Termination criterion:
      # If top-(m-1) blocks contain >= k values larger than
      # the m-th largest value, then top-k is guaranteed to be in top-(m-1)
      pivot = topk_vals[target_m - 1].max(-1, keepdims=True)
      num_larger = (
          sum([(v >= pivot) for v in topk_vals[:target_m - 1]])
          .astype(jnp.float32)
          .sum(-1)
      )

      termination_flag_ref[0] = 0
      for i in range(block_size):
        contains_topk = num_larger[i] >= k
        termination_flag_ref[0] += contains_topk

        # Record depth when criterion was met
        token_idx = pl.program_id(0) * block_size + i
        current_max = max_depth_ref[token_idx]
        max_depth_ref[token_idx] = jnp.where(
            contains_topk & (current_max == k),
            target_m - 1,
            current_max
        )

      # Check if all tokens converged
      @pl.when(termination_flag_ref[0] != block_size)
      def _():
        termination_flag_ref[0] = 0

  # Final top-k extraction (done by last program)
  @pl.when(pl.program_id(0) == (pl.num_programs(0) - 1))
  def _():
    # Find maximum depth across all tokens
    max_depth = jnp.array(0)
    for i in range(max_depth_ref.shape[0]):
      max_depth = jnp.maximum(max_depth, max_depth_ref[i])
    # convert to global indices from local
    block_topm_indices_ref[...] = (
        block_topm_indices_ref[...] * NUM_LANES
    ) + (
        jax.lax.broadcasted_iota(
            jnp.int32,
            block_topm_indices_ref.shape,
            1
        ) % NUM_LANES
    )

    # Use appropriate sorting depth based on max_depth
    for depth_lower, depth_upper in zip(topk_schedule, topk_schedule[1:]):
      @pl.when((max_depth > depth_lower) & (max_depth <= depth_upper))
      def _():
        # Sort the blockwise superset
        bitonic_sort(
            [ref.at[:, :depth_upper * NUM_LANES]
            for ref in (block_topm_values_ref, block_topm_indices_ref)],
            stage_ref=None,
            # this is a trick to make the sort descending
            dim1_offset=depth_upper * NUM_LANES,
            num_keys=1,
        )
        for ref, out_ref in zip(
          (block_topm_values_ref, block_topm_indices_ref),
          (topk_values_ref, topk_indices_ref)):
          out_ref[...] = ref[...,:out_ref.shape[1]].astype(out_ref.dtype)



@functools.partial(
    jit,
    static_argnames=("k", "block_size", "block_topk_schedule", "topk_schedule"),
)
def topk_pallas(
    logits,
    k: int,
    block_size: int = 8,
    block_topk_schedule=None,
    topk_schedule=None,
):
  """
  High-level interface for adaptive blockwise top-k on TPU.

  Args:
      logits: Input logits [num_tokens, vocab_size]
      k: Number of top elements to find
      block_size: Token blocking size
      block_topk_schedule: Schedule of m values for blockwise top-m
      topk_schedule: Schedule for final sorting depth

  Returns:
      Tuple of (values, indices) for top-k elements
  """
  num_tokens, vocab_size = logits.shape

  if num_tokens % block_size != 0:
    raise ValueError("num_tokens must be divisible by block_size")

  if topk_schedule is None:
    topk_schedule = (0, 8, k)

  if k > NUM_LANES:
    raise ValueError(f"k cannot exceed {NUM_LANES}")

  output_shapes = (
      jax.ShapeDtypeStruct((num_tokens, NUM_LANES), logits.dtype),
      jax.ShapeDtypeStruct((num_tokens, NUM_LANES), jnp.int32),
      jax.ShapeDtypeStruct((num_tokens,), jnp.int32),
  )

  output_specs = (
      pl.BlockSpec(),
      pl.BlockSpec(),
      pl.BlockSpec(memory_space=pltpu.SMEM),
  )

  topk_vals, topk_idxs, depths = pl.pallas_call(
      functools.partial(
          topk_blockwise_superset_kernel,
          k=k,
          block_topk_schedule=block_topk_schedule,
          topk_schedule=topk_schedule,
      ),
      in_specs=(
          pl.BlockSpec((block_size, vocab_size), lambda i: (i, 0)),
      ),
      out_shape=output_shapes,
      scratch_shapes=(
          pltpu.VMEM((num_tokens, k * NUM_LANES), jnp.float32),
          pltpu.VMEM((num_tokens, k * NUM_LANES), jnp.int32),
          pltpu.SMEM((1,), jnp.int32),
      ),
      grid=(num_tokens // block_size,),
      out_specs=output_specs,
      compiler_params=pltpu.CompilerParams(
        vmem_limit_bytes=int(0.9 * 2**27)
      ),
  )(logits)

  return topk_vals[:, :k], topk_idxs[:, :k]
