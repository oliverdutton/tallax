import functools
import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.tax.sort import bitonic_sort
from tallax.utils import unrolled_fori_loop, NUM_LANES, is_cpu_platform


def blockwise_topk(
    logits,
    max_k: int,
    block_topk_values,
    block_topk_indices,
    start_k: int = 0,
    num_blocks: int = NUM_LANES,
):
  """
  Compute blockwise top-k using a sinking sort approach.
  
  Args:
      logits: Input logits [num_tokens, num_blocks] (or similar slice)
      max_k: Static integer loop bound (maximum possible k)
      block_topk_values: Required pre-allocated buffers for values
      block_topk_indices: Required pre-allocated buffers for indices
      start_k: Starting position (for incremental top-k)
      num_blocks: Number of blocks to process
  """
  num_tokens = logits.shape[0]

  def process_block(block_idx, carry):
    """Process a single tile with sinking sort."""
    values_list, indices_list = carry

    # Extract current block
    current_values = logits[..., pl.dslice(num_blocks * block_idx, num_blocks)]
    current_indices = jnp.full((num_tokens, num_blocks), block_idx, jnp.int32)

    # Sinking sort: compare and swap through max_k levels
    for level in range(max_k):
      if level < start_k:
        # Invalidate already-found elements
        # We use the indices list to check identity
        current_values = jnp.where(
            current_indices == indices_list[level],
            float("-inf"),
            current_values
        )
      else:
        # Exchange with stored top-k
        # Only perform the swap if the value is larger
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

  return unrolled_fori_loop(
      logits.shape[-1] // num_blocks,
      process_block,
      (block_topk_values, block_topk_indices),
      unroll=16,
  )


def topk_blockwise_superset_kernel(
    logits_ref,
    k_ref,
    topk_values_ref,
    topk_indices_ref,
    max_depth_ref,
    block_topm_values_ref,
    block_topm_indices_ref,
    termination_flag_ref,
    *,
    max_k: int,
    block_topk_schedule: tuple[int],
    topk_schedule: tuple[int],
):
  """
  Compute blockwise top-k supersets until global top-k is guaranteed.
  
  Accepts dynamic k per row (k_ref) and uses static max_k for scheduling.
  """
  # Initialize buffers
  block_size = logits_ref.shape[0]
  shape = (block_size, block_topm_values_ref.shape[1])

  pid = pl.program_id(0)
  token_slice = pl.dslice(pid * block_size, block_size)
  
  block_topm_values_ref[token_slice] = jnp.full(
      shape, jnp.finfo(jnp.float32).min, dtype=jnp.float32
  )
  block_topm_indices_ref[token_slice] = jnp.full(shape, 0, dtype=jnp.int32)
  for i in range(block_size): 
    max_depth_ref[pid * block_size + i] = max_k
  termination_flag_ref[0] = 0

  # Incremental blockwise top-k computation
  for completed_m, target_m in zip(block_topk_schedule, block_topk_schedule[1:]):

    @pl.when(termination_flag_ref[0] == 0)
    def _():
      # Compute blockwise top-m
      topk_vals, topk_idxs = blockwise_topk(
          logits_ref,
          max_k=target_m,
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
          num_blocks=NUM_LANES,
          start_k=completed_m,
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
        token_idx = pid * block_size + i
        # Dynamic check against k
        contains_topk = num_larger[i] >= k_ref[token_idx]
        termination_flag_ref[0] += contains_topk

        # Record depth when criterion was met
        current_max = max_depth_ref[token_idx]
        max_depth_ref[token_idx] = jnp.where(
            contains_topk & (current_max == max_k),
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
    max_depth_global = jnp.array(0)
    for i in range(max_depth_ref.shape[0]):
      max_depth_global = jnp.maximum(max_depth_global, max_depth_ref[i])
      
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

    # Use appropriate sorting depth based on max_depth_global
    for depth_lower, depth_upper in zip(topk_schedule, topk_schedule[1:]):
      @pl.when((max_depth_global > depth_lower) & (max_depth_global <= depth_upper))
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
    static_argnames=("max_k", "block_size", "block_topk_schedule", "topk_schedule", "interpret"),
)
def top_dynamic_k(
    logits,
    k,
    max_k: int,
    block_size: int = 8,
    block_topk_schedule = None,
    topk_schedule = None,
    interpret: bool = False,
):
  """
  High-level interface for adaptive blockwise top-k on TPU.

  Args:
      logits: Input logits [num_tokens, vocab_size]
      k: JAX Array [num_tokens] containing k per row.
      max_k: Static integer maximum k (used for buffer sizing and compilation).
      block_size: Token blocking size.
      block_topk_schedule: Schedule of m values for blockwise top-m.
      topk_schedule: Schedule for final sorting depth.

  Returns:
      Tuple of (values, indices) for top-k elements.
      Output shape is fixed at [num_tokens, max_k].
  """
  num_tokens, vocab_size = logits.shape

  if num_tokens % block_size != 0:
    raise ValueError("num_tokens must be divisible by block_size")

  k = jnp.broadcast_to(k, (num_tokens,))

  if topk_schedule is None:
    topk_schedule = (8, max_k)
  topk_schedule = (0,) + topk_schedule
 
  if block_topk_schedule is None:
    block_topk_schedule = (5, 7, 9, 12, max_k)
  block_topk_schedule = (0,) + block_topk_schedule

  if topk_schedule[-1] < block_topk_schedule[-1]:
    raise ValueError('Top k max must cover block top m search')
    
  max_block_k_search = block_topk_schedule[-1]

  if max_k > NUM_LANES:
    raise ValueError(f"max_k cannot exceed {NUM_LANES}")

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
          max_k=max_k,
          block_topk_schedule=block_topk_schedule,
          topk_schedule=topk_schedule,
      ),
      in_specs=(
          pl.BlockSpec((block_size, vocab_size), lambda i: (i, 0)),
          pl.BlockSpec(memory_space=pltpu.SMEM),
      ),
      out_shape=output_shapes,
      scratch_shapes=(
          pltpu.VMEM((num_tokens, max_block_k_search * NUM_LANES), jnp.float32),
          pltpu.VMEM((num_tokens, max_block_k_search * NUM_LANES), jnp.int32),
          pltpu.SMEM((1,), jnp.int32),
      ),
      grid=(num_tokens // block_size,),
      out_specs=output_specs,
      compiler_params=pltpu.CompilerParams(
        vmem_limit_bytes=int(0.9 * 2**27)
      ),
      interpret=interpret,
  )(logits, k)
  
  if max_block_k_search == max_k:
    # must have converged
    valid = jnp.ones(num_tokens // block_size, dtype=bool)
  else:
    valid = (depths.reshape(-1, block_size) < max_k).all(1)
    
  return topk_vals[:,:max_k], topk_idxs[:,:max_k], valid

  
@functools.partial(
    jit,
    static_argnames=("k", "block_size", "block_topk_schedule", "topk_schedule", "interpret"),
)
def top_k(
    logits,
    k: int,
    block_size: int = 8,
    block_topk_schedule = None,
    topk_schedule = None,
    interpret: bool = False,
):
  return top_dynamic_k(
    logits,
    k=jnp.full(logits.shape[:1], k, dtype=jnp.int32),
    max_k=k,
    block_size=block_size,
    block_topk_schedule=block_topk_schedule,
    topk_schedule=topk_schedule,
    interpret=interpret,
  )[:2]
