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

from collections import OrderedDict
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
from tallax.tax.utils import NUM_LANES, map_reduce

_SAMPLING_EPS = 1e-5


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
  target_u48 = U48.from_u64_in_u32s(sampled_u64_in_u32s)
  return find_boundary_idx(
    unnorm_probs_i32,
    map_fn=lambda x: U48(x, max_val=max_val),
    target=target_u48,
  )


def topk_topp_mask_and_sample_kernel(
  logits_ref,
  random_u128_in_u32s_ref,
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
      return topk_topp_mask_and_sample_kernel(
        scoped_ref,
        random_u128_in_u32s_ref,
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
  vocab_size = logits_ref.shape[1]
  logits_max = map_reduce(logits_ref, reduce_fn="max")
  logits_max_lanes = jnp.broadcast_to(logits_max, (batch_size, NUM_LANES))

  # Stage 1: Greedy argmax
  greedy_sampled = find_boundary_idx(
    logits_ref,
    map_fn=lambda chunk: (
      chunk == pltpu.repeat(logits_max_lanes, chunk.shape[1] // NUM_LANES, 1)
    ).astype(jnp.int32),
    target=jnp.broadcast_to(jnp.float32(1), logits_max_lanes.shape),
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
  topk_logits /= temperature
  logits_max /= temperature

  # Stage 4: Top-p masking
  unnorm_probs_i32 = topp_mask(
    topk_logits,
    p_ref[...],
    logits_max=logits_max,
  )

  # Stage 5: Sample
  random_u128_parts = [ref[...] for ref in random_u128_in_u32s_ref]
  total_sum_u48 = U48.map_reduce_sum(unnorm_probs_i32, max_val=2**24 - 1)
  sampled_u64_in_u32s = modulo_u128_u64(
    random_u128_parts,
    total_sum_u48.to_u64_in_u32s(),
  )
  random_unnorm_cdf_sampled = U48.from_u64_in_u32s(sampled_u64_in_u32s)
  next_tokens = find_boundary_idx(
    unnorm_probs_i32,
    map_fn=lambda x: U48(x, max_val=2**24 - 1),
    target=random_unnorm_cdf_sampled,
  )

  sampled_tokens_ref[...] = jnp.where(
    temperature < _SAMPLING_EPS, greedy_sampled, next_tokens
  )

  # Debug: write full intermediate arrays if debug refs are provided
  if debug_arrays_ref is not None:
    debug_arrays_ref["greedy_sampled"][...] = greedy_sampled
    debug_arrays_ref["topk_logits_unsorted"][...] = topk_logits
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
def topk_topp_mask_and_sample(
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

  k = jnp.atleast_1d(k)
  p = jnp.atleast_1d(p)
  temperature = jnp.atleast_1d(temperature)

  if k.shape[0] == 1:
    k = jnp.broadcast_to(k, (batch_size,))
  if p.shape[0] == 1:
    p = jnp.broadcast_to(p, (batch_size,))
  if temperature.shape[0] == 1:
    temperature = jnp.broadcast_to(temperature, (batch_size,))

  k = jnp.reshape(k, (batch_size, 1))
  p = jnp.reshape(p, (batch_size, 1))
  temperature = jnp.reshape(temperature, (batch_size, 1))

  # Pad batch to multiple of block_token
  num_blocks = pl.cdiv(batch_size, block_token)
  padded_batch = num_blocks * block_token

  if padded_batch != batch_size:
    pad_size = padded_batch - batch_size
    logits = jnp.pad(
      logits, ((0, pad_size), (0, 0)), constant_values=replace_val
    )
    k = jnp.pad(k, ((0, pad_size), (0, 0)), constant_values=1)
    p = jnp.pad(p, ((0, pad_size), (0, 0)), constant_values=1.0)
    temperature = jnp.pad(
      temperature, ((0, pad_size), (0, 0)), constant_values=1.0
    )

  random_u128_in_u32s = tuple(
    sample_random_u128_in_u32s(rng_key, (padded_batch, 1))
  )

  output_shape = jax.ShapeDtypeStruct((padded_batch, 1), jnp.int32)

  if not debug:
    result = pl.pallas_call(
      functools.partial(
        topk_topp_mask_and_sample_kernel,
        stable=stable,
        replace_val=replace_val,
      ),
      out_shape=output_shape,
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
      out_specs=pl.BlockSpec((block_token, 1), lambda i: (i, 0)),
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 2**27)),
      interpret=interpret,
    )(logits, random_u128_in_u32s, k, p, temperature)

    return result[:batch_size, 0]
  else:
    # Debug mode: also output debug intermediates as full arrays
    debug_shapes = OrderedDict([
      ("greedy_sampled", jax.ShapeDtypeStruct((padded_batch, 1), jnp.int32)),
      (
        "topk_logits_unsorted",
        jax.ShapeDtypeStruct((padded_batch, vocab_size), jnp.float32),
      ),
      (
        "topk_topp_unnorm_probs_i32_unsorted",
        jax.ShapeDtypeStruct((padded_batch, vocab_size), jnp.int32),
      ),
      (
        "random_unnorm_cdf_sampled_high",
        jax.ShapeDtypeStruct((padded_batch, 1), jnp.uint32),
      ),
      (
        "random_unnorm_cdf_sampled_low",
        jax.ShapeDtypeStruct((padded_batch, 1), jnp.uint32),
      ),
      ("next_tokens", jax.ShapeDtypeStruct((padded_batch, 1), jnp.int32)),
    ])
    out_shapes = (output_shape, debug_shapes)
    result, debug_results = pl.pallas_call(
      functools.partial(
        topk_topp_mask_and_sample_kernel,
        stable=stable,
        replace_val=replace_val,
      ),
      out_shape=out_shapes,
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
        OrderedDict([
          ("greedy_sampled", pl.BlockSpec((block_token, 1), lambda i: (i, 0))),
          (
            "topk_logits_unsorted",
            pl.BlockSpec((block_token, vocab_size), lambda i: (i, 0)),
          ),
          (
            "topk_topp_unnorm_probs_i32_unsorted",
            pl.BlockSpec((block_token, vocab_size), lambda i: (i, 0)),
          ),
          (
            "random_unnorm_cdf_sampled_high",
            pl.BlockSpec((block_token, 1), lambda i: (i, 0)),
          ),
          (
            "random_unnorm_cdf_sampled_low",
            pl.BlockSpec((block_token, 1), lambda i: (i, 0)),
          ),
          ("next_tokens", pl.BlockSpec((block_token, 1), lambda i: (i, 0))),
        ]),
      ),
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 2**27)),
      interpret=interpret,
    )(logits, random_u128_in_u32s, k, p, temperature)

    # Trim padding and reshape debug outputs to match reference format
    debug_results_trimmed = OrderedDict([
      ("greedy_sampled", debug_results["greedy_sampled"][:batch_size, 0]),
      (
        "topk_logits_unsorted",
        debug_results["topk_logits_unsorted"][:batch_size, :],
      ),
      (
        "topk_topp_unnorm_probs_i32_unsorted",
        debug_results["topk_topp_unnorm_probs_i32_unsorted"][:batch_size, :],
      ),
      (
        "random_unnorm_cdf_sampled",
        (
          debug_results["random_unnorm_cdf_sampled_high"][:batch_size, 0],
          debug_results["random_unnorm_cdf_sampled_low"][:batch_size, 0],
        ),
      ),
      ("next_tokens", debug_results["next_tokens"][:batch_size, 0]),
    ])
    return result[:batch_size, 0], debug_results_trimmed
