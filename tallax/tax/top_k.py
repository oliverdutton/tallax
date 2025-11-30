
import functools
import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.tax.sort import bitonic_sort
from tallax.tax.topk_theory import calculate_depth_thresholds
from tallax.utils import unrolled_fori_loop, NUM_LANES, NUM_SUBLANES, pad, log2, get_dtype_info


def binned_topk(
    logits,
    k: int,
    bins_topk_vals,
    bins_topk_idxs,
    completed_k: int = 0,
    num_bins: int = NUM_LANES,
    unroll: int = 32,
):
  """
  Compute binned top-k using a sinking sort approach.
  
  Processes the vocabulary in num_bins-sized chunks, maintaining the top-k elements
  across all processed bins using a sinking sort algorithm. Values "sink" through
  the maintained top-k list if they are smaller than existing elements.
  
  Args:
      logits: Input logits of shape [num_tokens, vocab_size].
      k: Number of top elements to find.
      bins_topk_vals: List of k arrays, each of shape [num_tokens, num_bins],
          containing current top-k values per bin.
      bins_topk_idxs: List of k arrays, each of shape [num_tokens, num_bins],
          containing current top-k indices per bin.
      completed_k: Number of top-k positions already finalized (default: 0).
      num_bins: Number of bins/lanes to process simultaneously (default: 128).
      unroll: Loop unroll factor for the vocabulary scan (default: 32).
  
  Returns:
      Tuple of (bins_topk_vals, bins_topk_idxs) with updated top-k values and indices.
  """
  num_tokens, vocab_size = logits.shape

  def update_bins_topk(bubble_vals, bubble_idxs, bins_topk_vals, bins_topk_idxs):
    """
    Update bins topk with bubble vals/idxs using sinking sort.
    
    Compares new values against existing top-k, swapping when new values are larger.
    Already-completed positions are invalidated to prevent re-selection.
    """
    # Sinking sort: compare and swap
    for i in range(completed_k):
      # Invalidate already-found elements
      # We use the idxs list to check identity
      bubble_vals = jnp.where(
          bubble_idxs == bins_topk_idxs[i],
          jnp.finfo(jnp.float32).min,
          bubble_vals
      )
    for i in range(completed_k, k):
      # Exchange with stored top-k
      # Only perform the swap if the value is larger
      mask = bubble_vals > bins_topk_vals[i]
      bins_topk_vals[i], bubble_vals = (
          jnp.where(m, bubble_vals, bins_topk_vals[i])
          for m in (mask, ~mask)
      )
      bins_topk_idxs[i], bubble_idxs = (
          jnp.where(m, bubble_idxs, bins_topk_idxs[i])
          for m in (mask, ~mask)
      )
    return (bins_topk_vals, bins_topk_idxs)
  
  def compute_idxs(i):
    """Compute global vocabulary indices for bin slice i."""
    shape = (num_tokens, num_bins)
    return (
      jnp.full(shape, i * num_bins, jnp.int32) + 
      jax.lax.broadcasted_iota(jnp.int32, shape, 1)) 

  def loop_body(i, bins_topk_outs):
    vals = logits[..., pl.dslice(num_bins * i, num_bins)]
    idxs = compute_idxs(i)
    return update_bins_topk(vals, idxs, *bins_topk_outs)

  num_full_slices = vocab_size // num_bins
  bins_topk_outs = unrolled_fori_loop(
      num_full_slices,
      loop_body,
      (bins_topk_vals, bins_topk_idxs),
      unroll=unroll,
  )

  # Handle remaining elements if vocab_size doesn't divide num_bins
  remainder = vocab_size % num_bins
  if remainder > 0:
    # Load the final boundary segment
    final_vals = logits[..., pl.dslice(num_full_slices * num_bins, remainder)]
    # Pad to num_bins with f32 min
    final_vals = pad(final_vals, (1, num_bins), val=get_dtype_info(final_vals).min)
    # Create idxs for the final segment
    final_idxs = compute_idxs(num_full_slices)
    # Update bins topk with the overspill
    bins_topk_outs = update_bins_topk(final_vals, final_idxs, *bins_topk_outs)
  return bins_topk_outs


def dynamic_topk_kernel(
    logits_ref,
    k_ref,
    topk_vals_ref,
    topk_idxs_ref,
    valid_ref,
    max_depth_ref,
    cutoff_vals_ref,
    bins_topm_vals_ref,
    bins_topm_idxs_ref,
    termination_flag_ref,
    sorted_bins_ref,
    packed_data_vals_ref,
    packed_data_idxs_ref,
    *,
    max_k: int,
    num_bins: int,
    bins_topm_unroll: int,
    bins_topm_schedule: tuple[int, ...],
    enable_bin_sorting: bool,
):
  """
  Pallas kernel for computing binned top-k supersets until global top-k is guaranteed.
  
  Incrementally computes top-m supersets (m increasing per schedule) until the top-k
  is provably contained within the top-(m-1) bins. Supports dynamic k per token while
  using static max_k for compilation and scheduling.
  
  The termination criterion checks if the top-(m-1) bins collectively contain at least
  k values larger than the largest m-th largest value across all bins. 
  """
  # Initialize buffers
  block_token = logits_ref.shape[0]
  shape = (block_token, bins_topm_vals_ref.shape[1])

  pid = pl.program_id(0)
  token_slice = pl.dslice(pid * block_token, block_token)

  bins_topm_vals_ref[token_slice] = jnp.full(
      shape, jnp.finfo(jnp.float32).min, dtype=jnp.float32
  )

  for i in range(block_token): 
    max_depth_ref[pid * block_token + i] = max_k
  termination_flag_ref[0] = 0

  # Incremental binned top-k computation
  for completed_m, m in zip(bins_topm_schedule, bins_topm_schedule[1:]):
    @pl.when(termination_flag_ref[0] == 0)
    def _():
      # Compute binned top-m
      bins_topm_vals, bins_topm_idxs = binned_topk(
          logits_ref,
          k=m,
          bins_topk_vals=[
              bins_topm_vals_ref[
                  token_slice, pl.dslice(i * num_bins, num_bins)
              ].astype(jnp.float32)
              for i in range(m)
          ],
          bins_topk_idxs=[
              bins_topm_idxs_ref[
                  token_slice, pl.dslice(i * num_bins, num_bins)
              ]
              for i in range(m)
          ],
          num_bins=num_bins,
          completed_k=completed_m,
          unroll=bins_topm_unroll,
      )

      # Store results
      for i in range(completed_m, m):
        bins_topm_vals_ref[
            token_slice, pl.dslice(i * num_bins, num_bins)
        ] = bins_topm_vals[i].astype(bins_topm_vals_ref.dtype)
        bins_topm_idxs_ref[
            token_slice, pl.dslice(i * num_bins, num_bins)
        ] = bins_topm_idxs[i].astype(bins_topm_idxs_ref.dtype)

      # Termination criterion:
      # If top-(m-1) bins contain >= k vals larger than
      # the largest m-th largest value, then top-k is guaranteed to be in bins 
      # top-(m-1) collated
      pivot = bins_topm_vals[m - 1].max(-1, keepdims=True)
      num_larger = (
          sum([(v >= pivot) for v in bins_topm_vals[:m - 1]])
          .astype(jnp.float32)
          .sum(-1)
      )

      termination_flag_ref[0] = 0
      for i in range(block_token):
        token_idx = pid * block_token + i
        # Dynamic check against k
        contains_topk = num_larger[i] >= k_ref[token_idx]
        termination_flag_ref[0] += contains_topk

        # Record depth when criterion was met
        current_max = max_depth_ref[token_idx]
        max_depth_ref[token_idx] = jnp.where(
            contains_topk & (current_max == max_k),
            m - 1,
            current_max
        )
        # Record largest m-th largest value
        # Useful for bounds checking if running sharded topk
        cutoff_vals_ref[token_idx] = pivot.squeeze(1)[i]

      # Check if all tokens converged
      @pl.when(termination_flag_ref[0] != block_token)
      def _():
        termination_flag_ref[0] = 0

  # Bin sorting optimization for rare non-convergence cases
  @pl.when(enable_bin_sorting & (termination_flag_ref[0] == 0))
  def _():
    # This optimization applies when:
    # - bin_schedule = (5,9), k=128, num_bins=256
    # - Not converged at iteration 9 (checking k up to 8)
    # - At most 16 bins contain values contributing to top-k

    m = bins_topm_schedule[-1]  # Should be 9 for the (5,9) schedule

    # Assertion: bins_topm_schedule should be (0, 5, 9) for this optimization
    assert len(bins_topm_schedule) == 3, "Expected schedule (0, 5, 9) for bin sorting"
    assert bins_topm_schedule[1] == 5 and bins_topm_schedule[2] == 9, \
        "Expected schedule (0, 5, 9) for bin sorting"

    # Count contribution of each bin to top-k
    # bins_topm_vals has shape (m, block_token, num_bins)
    # We want to count how many values in each bin are >= pivot
    pivot = bins_topm_vals_ref[token_slice, pl.dslice((m - 1) * num_bins, num_bins)].max(-1, keepdims=True)

    # Count contributions per bin across the m-1 top bins
    # Shape: (block_token, num_bins)
    num_gt_k = jnp.zeros((block_token, num_bins), dtype=jnp.int32)
    for i in range(m - 1):
      bin_vals = bins_topm_vals_ref[token_slice, pl.dslice(i * num_bins, num_bins)]
      num_gt_k += (bin_vals >= pivot).astype(jnp.int32)

    # Assertion: at most 16 bins should have contributions (num_gt_k > 0)
    num_active_bins = (num_gt_k > 0).sum(-1)
    for i in range(block_token):
      assert num_active_bins[i] <= 16, \
          f"Expected at most 16 active bins, got {num_active_bins[i]}"

    # Use jax argsort descending to get bin indices ordered by contribution count
    # argsort gives indices that would sort the array (ascending by default)
    # For descending, we negate the values
    argsort_indices = jnp.argsort(-num_gt_k, axis=1)

    # Extract top NUM_LANES (128) bin indices
    # Shape: (block_token, NUM_LANES)
    permutation = argsort_indices[:, :NUM_LANES]

    # Store sorted bin indices for reference
    sorted_bins_ref[token_slice] = permutation.astype(sorted_bins_ref.dtype)

    # Make permutation repeat first 16 values using modulo
    # Create 2D iota and take modulo 16
    iota_2d = jax.lax.broadcasted_iota(jnp.int32, (block_token, NUM_LANES), 1)
    repeated_indices = iota_2d % 16
    permutation_repeated = jnp.take_along_axis(
        permutation, repeated_indices, axis=1
    )

    # Process blocks of data
    num_blocks = num_bins
    # Output will have shape (block_token, NUM_LANES) after packing
    # Initialize output with -inf
    packed_vals = jnp.full(
        (block_token, NUM_LANES),
        jnp.finfo(jnp.float32).min,
        dtype=jnp.float32
    )
    packed_idxs = jnp.zeros((block_token, NUM_LANES), dtype=jnp.int32)

    # Loop over blocks in chunks of NUM_LANES
    for offset in range(0, num_blocks, NUM_LANES):
      # Tile permutation by subtracting offset
      tile_permutation = permutation_repeated - offset

      # Mask for valid indices in this block
      in_range_mask = (tile_permutation >= 0) & (tile_permutation < NUM_LANES)

      # Clip tile_permutation to valid range for take_along_axis
      tile_permutation_clipped = jnp.clip(tile_permutation, 0, NUM_LANES - 1)

      # For each m-1 bin level, gather the data
      for bin_level in range(m - 1):
        # Get the logit values for this bin level
        bin_vals = bins_topm_vals_ref[
            token_slice, pl.dslice(bin_level * num_bins + offset, NUM_LANES)
        ]
        bin_idxs = bins_topm_idxs_ref[
            token_slice, pl.dslice(bin_level * num_bins + offset, NUM_LANES)
        ]

        # Permute according to tile_permutation
        # Clip is safe to use because we mask out-of-range values anyway
        permuted_vals = jnp.take_along_axis(bin_vals, tile_permutation_clipped, axis=1)
        permuted_idxs = jnp.take_along_axis(bin_idxs, tile_permutation_clipped, axis=1)

        # Only keep valid values (mask out-of-range)
        permuted_vals = jnp.where(in_range_mask, permuted_vals, jnp.finfo(jnp.float32).min)
        permuted_idxs = jnp.where(in_range_mask, permuted_idxs, 0)

        # Pack into output: first 16 positions get bin_level 0,
        # next 16 get bin_level 1, etc.
        # Create mask for this bin level's positions
        start_pos = bin_level * 16
        end_pos = (bin_level + 1) * 16
        pack_mask = (iota_2d >= start_pos) & (iota_2d < end_pos) & in_range_mask

        # Update packed data where mask is true
        packed_vals = jnp.where(pack_mask, permuted_vals, packed_vals)
        packed_idxs = jnp.where(pack_mask, permuted_idxs, packed_idxs)

    # Write packed data to output refs
    packed_data_vals_ref[token_slice] = packed_vals.astype(packed_data_vals_ref.dtype)
    packed_data_idxs_ref[token_slice] = packed_idxs.astype(packed_data_idxs_ref.dtype)

  global_topk_schedule = tuple(sorted(set(2**log2(x - 1) if x >1 else x for x in bins_topm_schedule)))

  # Final top-k extraction (done by last program)
  @pl.when(pl.program_id(0) == (pl.num_programs(0) - 1))
  def _():
    # Find maximum depth across all tokens
    global_max_depth = jnp.array(0)
    for i in range(max_depth_ref.shape[0]):
      global_max_depth = jnp.maximum(global_max_depth, max_depth_ref[i])
    
    valid_ref[0] = ((
    global_max_depth < bins_topm_schedule[-1]
    ) | (bins_topm_schedule[-1] == max_k)
    ).astype(jnp.int32)

    # Use appropriate sorting depth based on global_max_depth
    for depth_lower, depth_upper in zip(global_topk_schedule, global_topk_schedule[1:]):
      @pl.when((
      (global_max_depth > depth_lower) & (global_max_depth <= depth_upper)
      ) | (
      # Sort to give approx topk if not fully converged
      (depth_upper == global_topk_schedule[-1]) & (global_max_depth > depth_upper)
      ))
      def _():
        # Sort the binned superset
        bitonic_sort(
            [ref.at[:, :depth_upper * num_bins]
            for ref in (bins_topm_vals_ref, bins_topm_idxs_ref)],
            stage_ref=None,
            # this is a trick to make the sort descending
            dim1_offset=depth_upper * num_bins,
            num_keys=1,
        )
        for ref, out_ref in zip(
          (bins_topm_vals_ref, bins_topm_idxs_ref),
          (topk_vals_ref, topk_idxs_ref)):
          out_ref[...] = ref[...,:out_ref.shape[1]].astype(out_ref.dtype)


@functools.partial(
    jit,
    static_argnames=(
        "max_k",
        "block_token",
        "num_bins",
        "bins_topm_unroll",
        "bins_topm_schedule",
        "guarantee_convergence",
        "enable_bin_sorting",
        "interpret"
    ),
)
def top_dynamic_k(
    logits,
    k,
    max_k: int,
    block_token: int = 8,
    num_bins: int = NUM_LANES,
    bins_topm_unroll: int = 32,
    bins_topm_schedule: tuple[int, ...] | None = None,
    guarantee_convergence: bool = False,
    enable_bin_sorting: bool = False,
    interpret: bool = False,
):
  """
  High-level interface for adaptive binned top-k computation on TPU.

  Supports dynamic k per token (each token can have a different k value) while
  maintaining efficient TPU execution through static compilation based on max_k.
  Automatically computes optimal search schedules if not provided.

  Args:
      logits: Input logits of shape [num_tokens, vocab_size].
      k: Per-token k values. Can be scalar (broadcast to all tokens) or array
          of shape [num_tokens].
      max_k: Static maximum k across all tokens. Used for buffer sizing and
          compilation. Must be >= all values in k.
      block_token: Number of tokens processed per program block (default: 8).
          Must evenly divide num_tokens.
      num_bins: Number of bins for parallel binned operations (default: 128).
      bins_topm_unroll: Loop unroll factor for binned top-m inner loop (default: 32).
      bins_topm_schedule: Increasing sequence of m values for incremental top-m search.
          If None, automatically computed based on convergence probability thresholds.
      guarantee_convergence: If True, adds max_k to schedule to ensure full convergence
          (default: False).
      enable_bin_sorting: If True, enables bin sorting optimization for rare non-convergence
          cases. Requires bins_topm_schedule=(0,5,9), k=128, num_bins=256 (default: False).
          Returns additional outputs when enabled.
      interpret: If True, run in CPU interpret mode instead of TPU compilation (default: False).

  Returns:
      When enable_bin_sorting=False:
          Tuple of (topk_vals, topk_idxs, valid, depths, cutoff_vals):
              - topk_vals: Top-k values of shape [num_tokens, max_k].
              - topk_idxs: Top-k indices of shape [num_tokens, max_k].
              - valid: Boolean indicating if algorithm fully converged.
              - depths: Per-token convergence depth of shape [num_tokens].
              - cutoff_vals: Per-token pivot values of shape [num_tokens].
      When enable_bin_sorting=True:
          Tuple of (topk_vals, topk_idxs, valid, depths, cutoff_vals, sorted_bins,
                    packed_data_vals, packed_data_idxs):
              - sorted_bins: Bin indices sorted by contribution count, shape [num_tokens, 128].
              - packed_data_vals: Packed values from top 16 bins, shape [num_tokens, 128].
              - packed_data_idxs: Packed indices from top 16 bins, shape [num_tokens, 128].
  """
  num_tokens, vocab_size = logits.shape

  if num_tokens % block_token != 0:
    raise ValueError("num_tokens must be divisible by block_token")

  k = jnp.broadcast_to(k, (num_tokens,))

  # Auto-compute schedules if not provided
  if bins_topm_schedule is None:
    thresholds = calculate_depth_thresholds(max_k, num_bins, block_token, target_yields=(0.8, 0.98, 0.9999))
    bins_topm_schedule = tuple(t + 1 for t in thresholds)
    print(f"Auto-computed schedules for max_k={max_k}, num_bins={num_bins}:")
    print(f"  bins_topm_schedule: {bins_topm_schedule}")
  if guarantee_convergence:
    bins_topm_schedule += (max_k,)
  bins_topm_schedule = tuple(sorted(set(bins_topm_schedule)))
  bins_topm_schedule = (0,) + bins_topm_schedule
  # binned topk / sort pad len
  max_m = bins_topm_schedule[-1]
  buffer_size = max(max_m, 2**log2(max_m - 1)) * num_bins

  # Updated padded size calculation using num_bins
  padded_max_k = pl.cdiv(max_k, NUM_LANES) * NUM_LANES

  output_shapes = (
      jax.ShapeDtypeStruct((num_tokens, padded_max_k), logits.dtype),
      jax.ShapeDtypeStruct((num_tokens, padded_max_k), jnp.int32),
      jax.ShapeDtypeStruct((1,), jnp.int32),
      jax.ShapeDtypeStruct((num_tokens,), jnp.int32),
      jax.ShapeDtypeStruct((num_tokens,), jnp.float32),
      jax.ShapeDtypeStruct((num_tokens, NUM_LANES), jnp.int32) if enable_bin_sorting else jax.ShapeDtypeStruct((1,), jnp.int32),
      jax.ShapeDtypeStruct((num_tokens, NUM_LANES), logits.dtype) if enable_bin_sorting else jax.ShapeDtypeStruct((1,), logits.dtype),
      jax.ShapeDtypeStruct((num_tokens, NUM_LANES), jnp.int32) if enable_bin_sorting else jax.ShapeDtypeStruct((1,), jnp.int32),
  )

  output_specs = (
      pl.BlockSpec(),
      pl.BlockSpec(),
      pl.BlockSpec(memory_space=pltpu.SMEM),
      pl.BlockSpec(memory_space=pltpu.SMEM),
      pl.BlockSpec(memory_space=pltpu.SMEM),
      pl.BlockSpec((block_token, NUM_LANES), lambda i: (i, 0)) if enable_bin_sorting else pl.BlockSpec(memory_space=pltpu.SMEM),
      pl.BlockSpec((block_token, NUM_LANES), lambda i: (i, 0)) if enable_bin_sorting else pl.BlockSpec(memory_space=pltpu.SMEM),
      pl.BlockSpec((block_token, NUM_LANES), lambda i: (i, 0)) if enable_bin_sorting else pl.BlockSpec(memory_space=pltpu.SMEM),
  )

  results = pl.pallas_call(
      functools.partial(
          dynamic_topk_kernel,
          max_k=max_k,
          num_bins=num_bins,
          bins_topm_unroll=bins_topm_unroll,
          bins_topm_schedule=bins_topm_schedule,
          enable_bin_sorting=enable_bin_sorting,
      ),
      in_specs=(
          pl.BlockSpec((block_token, vocab_size), lambda i: (i, 0)),
          pl.BlockSpec(memory_space=pltpu.SMEM),
      ),
      out_shape=output_shapes,
      scratch_shapes=(
          pltpu.VMEM((num_tokens, buffer_size), jnp.float32),
          pltpu.VMEM((num_tokens, buffer_size), jnp.int32),
          pltpu.SMEM((1,), jnp.int32),
      ),
      grid=(num_tokens // block_token,),
      out_specs=output_specs,
      compiler_params=pltpu.CompilerParams(
        vmem_limit_bytes=int(0.9 * 2**27)
      ),
      interpret=interpret,
  )(logits, k)

  topk_vals, topk_idxs, valid, depths, cutoff_vals, sorted_bins, packed_data_vals, packed_data_idxs = results

  if enable_bin_sorting:
    return (topk_vals[:,:max_k], topk_idxs[:,:max_k], valid.squeeze().astype(bool),
            depths, cutoff_vals, sorted_bins, packed_data_vals, packed_data_idxs)
  else:
    return topk_vals[:,:max_k], topk_idxs[:,:max_k], valid.squeeze().astype(bool), depths, cutoff_vals

  
@functools.partial(
    jit,
    static_argnames=(
        "k", 
        "block_token", 
        "num_bins", 
        "bins_topm_unroll", 
        "bins_topm_schedule",
        "interpret"
    ),
)
def top_k(
    logits,
    k: int,
    block_token: int = NUM_SUBLANES,
    num_bins: int = NUM_LANES,
    bins_topm_unroll: int = 32,
    bins_topm_schedule: tuple[int, ...] | None = None,
    interpret: bool = False,
):
  """
  Compute top-k elements with guaranteed convergence.
  
  Simplified interface for uniform k across all tokens. Automatically ensures
  convergence by setting guarantee_convergence=True internally.
  
  Args:
      logits: Input logits of shape [num_tokens, vocab_size].
      k: Number of top elements to find (uniform across all tokens).
      block_token: Number of tokens processed per program block (default: 8).
      num_bins: Number of bins for parallel operations (default: 128).
      bins_topm_unroll: Loop unroll factor for inner loop (default: 32).
      bins_topm_schedule: Optional custom search schedule. If None, automatically
          computed.
      interpret: If True, run in CPU interpret mode (default: False).
  
  Returns:
      Tuple of (topk_vals, topk_idxs):
          - topk_vals: Top-k values of shape [num_tokens, k].
          - topk_idxs: Top-k indices of shape [num_tokens, k].
  """
  return top_dynamic_k(
    logits,
    k=k,
    max_k=k,
    block_token=block_token,
    num_bins=num_bins,
    bins_topm_unroll=bins_topm_unroll,
    bins_topm_schedule=bins_topm_schedule,
    guarantee_convergence=True,
    interpret=interpret,
  )[:2]
