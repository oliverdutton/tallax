"""Full-vocabulary top-k + top-p + sample Pallas kernel.

This kernel operates on the full vocabulary using binary search for both
top-k and top-p stages. It never sorts the input.

Pipeline:
  1. Greedy argmax via find_boundary_idx
  2. Top-k masking via binary search in f32 space
  3. Temperature scaling
  4. Top-p masking via binary search in i32 probability space
  5. Sampling via U48 cumsum + modulo_u128_u64

Optional debug_results: a nested dict of int32[1] SMEM values (1=match, 0=mismatch)
for verifying that each intermediate matches the reference implementation.
"""

import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.vllm.utils.high_precision_uint import (
  U48,
  modulo_u128_u64,
  sample_random_u128_in_u32s,
)
from tallax.vllm.arbitrary_k.topp_mask import topp_mask
from tallax.vllm.arbitrary_k.topk_mask import topk_mask, find_boundary_idx
from tallax.tax.utils import NUM_LANES, map_reduce, get_dtype_info

_SAMPLING_EPS = 1e-5


def canonicalize_args(batch_size, block_token, *args_with_pad):
  """Canonicalize per-sample args and pad batch to block_token multiple.

  Each arg is (value, pad_constant). 1D/scalar args are broadcast to batch_size
  and reshaped to (batch_size, 1). All args are padded to the next multiple
  of block_token.

  Returns (padded_batch, num_blocks, *padded_args)
  """
  num_blocks = pl.cdiv(batch_size, block_token)
  padded_batch = num_blocks * block_token
  pad_size = padded_batch - batch_size

  result = []
  for val, pad_val in args_with_pad:
    val = jnp.atleast_1d(val)
    if val.ndim == 1:
      if val.shape[0] == 1:
        val = jnp.broadcast_to(val, (batch_size,))
      val = jnp.reshape(val, (batch_size, 1))
    if pad_size > 0:
      val = jnp.pad(val, ((0, pad_size), (0, 0)), constant_values=pad_val)
    result.append(val)
  return (padded_batch, num_blocks, *result)


def sample_probs(unnorm_probs_i32, random_u128_in_u32s, max_val=2**24 - 1):
  """Sample from unnormalized i32 probabilities using U48 arithmetic.

  Args:
    unnorm_probs_i32: Unnormalized probabilities [batch, vocab_size] in i32
    random_u128_in_u32s: List of 4 u32 arrays for random sampling
    max_val: Maximum bound of unnorm_probs_i32 values

  Returns:
    Sampled token indices [batch, NUM_LANES]
  """
  total_sum_u48 = U48.map_reduce_sum(unnorm_probs_i32, max_val=max_val)
  sampled_u64_in_u32s = modulo_u128_u64(
    random_u128_in_u32s,
    total_sum_u48.to_u64_in_u32s(),
  )
  random_unnorm_cdf_sampled = U48.from_u64_in_u32s(sampled_u64_in_u32s)
  return find_boundary_idx(
    unnorm_probs_i32,
    map_fn=lambda x: U48(x, max_val=max_val),
    target=random_unnorm_cdf_sampled,
    pad_val=0,
  ), random_unnorm_cdf_sampled


def arbitrary_topk_topp_and_sample_kernel(
  logits_ref,
  random_u128_in_u32s_refs,
  k_ref,
  p_ref,
  temperature_ref,
  sampled_tokens_ref,
  debug_arrays_ref=None,
  *,
  stable: bool,
  replace_val: float,
  underlying_logits_dtype=None,
):
  """Pallas kernel body for combined topk + topp + sample.

  Args:
    logits_ref: Input logits [block_token, vocab_size]
    random_u128_in_u32s_ref: 4-tuple of [block_token, 1] u32 refs
    k_ref: Top-k values [block_token, 1]
    p_ref: Top-p values [block_token, 1]
    temperature_ref: Temperature values [block_token, 1]
    sampled_tokens_ref: Output sampled tokens [block_token, 1]
    debug_arrays_ref: Optional dict of refs for full intermediate arrays
    stable: Whether to use stable masking
    replace_val: Replacement value for masked elements
    underlying_logits_dtype: Original dtype if logits were cast
  """
  if logits_ref.dtype != jnp.float32:

    def scoped_body(scoped_ref):
      scoped_ref[...] = logits_ref[...].astype(jnp.float32)
      return arbitrary_topk_topp_and_sample_kernel(
        scoped_ref,
        random_u128_in_u32s_refs,
        k_ref,
        p_ref,
        temperature_ref,
        sampled_tokens_ref,
        debug_arrays_ref=debug_arrays_ref,
        stable=stable,
        replace_val=replace_val,
        underlying_logits_dtype=logits_ref.dtype,
      )

    return pl.run_scoped(scoped_body, pltpu.VMEM(logits_ref.shape, jnp.float32))

  batch_size = logits_ref.shape[0]
  logits_max = map_reduce(logits_ref, reduce_fn="max")
  logits_max_lanes = jnp.broadcast_to(logits_max, (batch_size, NUM_LANES))

  # Stage 1: Greedy argmax
  greedy_sampled = find_boundary_idx(
    logits_ref,
    map_fn=lambda chunk: (
      chunk == pltpu.repeat(logits_max_lanes, chunk.shape[1] // NUM_LANES, 1)
    ).astype(jnp.int32),
    target=jnp.broadcast_to(jnp.float32(1), logits_max_lanes.shape),
    pad_val=get_dtype_info(logits_ref).min,
  )[:, :1]

  # Stage 2: Top-k masking
  temperature = temperature_ref[...].astype(logits_ref.dtype)
  topk_logits = topk_mask(
    logits_ref,
    k_ref,
    replace_val=replace_val,
    stable=stable,
    underlying_dtype=underlying_logits_dtype,
  )

  # Stage 3: Temperature scaling
  topk_logits_pre_temperature = topk_logits
  topk_logits /= temperature
  logits_max /= temperature

  # Stage 4: Top-p masking
  unnorm_probs_i32 = topp_mask(
    topk_logits,
    p_ref[...],
    logits_max=logits_max,
  )

  # Stage 5: Sample
  next_tokens, random_unnorm_cdf_sampled = sample_probs(
    unnorm_probs_i32,
    [ref[...] for ref in random_u128_in_u32s_refs],
    max_val=2**24 - 1,
  )
  sampled_tokens_ref[...] = jnp.where(
    temperature < _SAMPLING_EPS, greedy_sampled, next_tokens
  )

  # Debug: write full intermediate arrays if debug refs are provided
  if debug_arrays_ref is not None:
    debug_arrays_ref["greedy_sampled"][...] = greedy_sampled
    ref = debug_arrays_ref["topk_logits_unsorted"]
    ref[...] = topk_logits_pre_temperature.astype(ref.dtype)
    debug_arrays_ref["topk_topp_unnorm_probs_i32_unsorted"][...] = (
      unnorm_probs_i32
    )
    # Store as tuple of (high_u32, low_u32)
    for ref, val in zip(
      debug_arrays_ref["random_unnorm_cdf_sampled"],
      random_unnorm_cdf_sampled.to_u64_in_u32s(),
      strict=True,
    ):
      ref[...] = val
    debug_arrays_ref["next_tokens"][...] = next_tokens


@functools.partial(
  jax.jit,
  static_argnames=[
    "stable",
    "replace_val",
    "block_token",
    "interpret",
    "debug",
  ],
)
def arbitrary_topk_topp_and_sample(
  logits: jax.Array,
  rng_key: jax.Array,
  k: jax.Array,
  p: jax.Array,
  temperature: jax.Array,
  *,
  stable: bool = True,
  replace_val: float = -1e12,
  block_token: int = 8,
  interpret: bool = False,
  debug: bool = False,
) -> jax.Array:
  """Full-vocabulary top-k + top-p masking and sampling via Pallas.

  This path keeps the full vocabulary and uses binary search for both
  top-k and top-p thresholds. Never sorts.

  Args:
    logits: Input logits [batch, vocab_size]
    rng_key: RNG key
    k: Top-k values [batch] or scalar
    p: Top-p values [batch] or scalar
    temperature: Temperature values [batch] or scalar
    stable: Whether to use stable masking
    replace_val: Replacement value for masked elements
    block_token: Number of tokens per block
    interpret: Whether to use interpret mode
    debug: If True, return (sampled_tokens, debug_results) dict

  Returns:
    Sampled token indices [batch], or (tokens, debug_dict) if debug=True
  """
  batch_size, vocab_size = logits.shape

  padded_batch, num_blocks, logits, k, p, temperature = canonicalize_args(
    batch_size,
    block_token,
    (logits, 0.0),
    (k, 1),
    (p, 1.0),
    (temperature, 1.0),
  )

  random_u128_in_u32s = tuple(
    sample_random_u128_in_u32s(rng_key, (padded_batch, 1))
  )

  output_shape = jax.ShapeDtypeStruct((padded_batch, 1), jnp.int32)

  if debug:
    debug_out_shapes = {
      "greedy_sampled": jax.ShapeDtypeStruct((padded_batch, 1), jnp.int32),
      "topk_logits_unsorted": jax.ShapeDtypeStruct(
        (padded_batch, vocab_size), logits.dtype
      ),
      "topk_topp_unnorm_probs_i32_unsorted": jax.ShapeDtypeStruct(
        (padded_batch, vocab_size), jnp.int32
      ),
      "random_unnorm_cdf_sampled": (
        jax.ShapeDtypeStruct((padded_batch, 1), jnp.uint32),
      )
      * 2,
      "next_tokens": jax.ShapeDtypeStruct((padded_batch, 1), jnp.int32),
    }
    debug_out_specs = {
      "greedy_sampled": pl.BlockSpec((block_token, 1), lambda i: (i, 0)),
      "topk_logits_unsorted": pl.BlockSpec(
        (block_token, vocab_size), lambda i: (i, 0)
      ),
      "topk_topp_unnorm_probs_i32_unsorted": pl.BlockSpec(
        (block_token, vocab_size), lambda i: (i, 0)
      ),
      "random_unnorm_cdf_sampled": (
        pl.BlockSpec((block_token, 1), lambda i: (i, 0)),
      )
      * 2,
      "next_tokens": pl.BlockSpec((block_token, 1), lambda i: (i, 0)),
    }
  else:
    debug_out_shapes = None
    debug_out_specs = None

  results = pl.pallas_call(
    functools.partial(
      arbitrary_topk_topp_and_sample_kernel,
      stable=stable,
      replace_val=replace_val,
    ),
    out_shape=(output_shape, debug_out_shapes),
    grid=(num_blocks,),
    in_specs=[
      pl.BlockSpec((block_token, vocab_size), lambda i: (i, 0)),
      jax.tree.map(
        lambda x: pl.BlockSpec((block_token, 1), lambda i: (i, 0)),
        random_u128_in_u32s,
      ),
      pl.BlockSpec((block_token, 1), lambda i: (i, 0)),
      pl.BlockSpec((block_token, 1), lambda i: (i, 0)),
      pl.BlockSpec((block_token, 1), lambda i: (i, 0)),
    ],
    out_specs=(
      pl.BlockSpec((block_token, 1), lambda i: (i, 0)),
      debug_out_specs,
    ),
    compiler_params=pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 2**27)),
    interpret=interpret,
  )(logits, random_u128_in_u32s, k, p, temperature)

  def trim(x):
    if x.shape[1] == 1:
      x = x.squeeze(1)
    return x[:batch_size]

  sampled_tokens, debug_aux = jax.tree.map(trim, results)

  if debug:
    return sampled_tokens, debug_aux
  return sampled_tokens
