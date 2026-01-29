"""Pallas kernel for combined top-k, top-p masking and sampling."""

import functools
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.tax.bitonic.topk import max_arrays
from tallax.vllm.high_precision_uint import U48, random_u48
from tallax.vllm.topp_mask import topp_mask, sum_in_u48
from tallax.vllm.topk_mask import topk_mask_ref_inputs, find_boundary_idx
from tallax.tax.sparse_random import sparse_random_categorical
from tallax.tax.utils import NUM_LANES, map_reduce

_SAMPLING_EPS = 1e-5


def topk_topp_mask_and_sample_kernel(
  logits_ref,
  rng_key_ref,
  k_ref,
  p_ref,
  temperature_ref,
  dim0_offset_ref,
  sampled_tokens_ref,
  *,
  stable: bool,
  replace_val: float,
  sample_in_i32: bool,
  underlying_logits_dtype = None,
):
  """Pallas kernel for topk/topp masking and sampling.

  Args:
    logits_ref: Input logits [block_token, vocab_size]
    rng_key_ref: RNG key [1, 2]
    k_ref: Top-k values [block_token, 1]
    p_ref: Top-p values [block_token, 1]
    temperature_ref: Temperature values [block_token, 1]
    dim0_offset_ref: Offset for batch indexing [1]
    sampled_tokens_ref: Output sampled tokens [block_token, 1]
    stable: Whether to use stable masking
    replace_val: Replacement value for masked elements
  """
  if logits_ref.dtype != jnp.float32:
    # We need a ref, as it's the only way to do dynamic_slices in mosaic
    # If it's not a ref (doesn't have memory space attr), we'll turn it into one.
    # We don't do this unconditionally as it incurs a copy
    def scoped_body(scoped_ref):
      scoped_ref[...] = logits_ref[...].astype(jnp.float32)
      return topk_topp_mask_and_sample_kernel(
        scoped_ref, rng_key_ref,  k_ref,
        p_ref,
        temperature_ref,
        dim0_offset_ref,
        sampled_tokens_ref,
        stable=stable,
        replace_val=replace_val,
        sample_in_i32=sample_in_i32,
        underlying_logits_dtype=logits_ref.dtype,
      )
    return pl.run_scoped(scoped_body, pltpu.VMEM(logits_ref.shape, jnp.float32))

  # logits = logits_ref[...]
  batch_size = logits_ref.shape[0]
  logits_max = map_reduce(
    logits_ref,
    reduce_fn="max",
  )
  logits_max_lanes = jnp.broadcast_to(logits_max, (batch_size, NUM_LANES))
  greedy_sampled = find_boundary_idx(
    logits_ref,
    map_fn=lambda chunk: (chunk == pltpu.repeat(logits_max_lanes, chunk.shape[1] // NUM_LANES, 1)).astype(jnp.int32),
    # stable -> first matching index
    target=jnp.broadcast_to(jnp.float32(1), logits_max_lanes.shape),
  )[:, :1]

  # Create token indices for greedy sampling and RNG seeding
  # token_idx = lax.broadcasted_iota(jnp.int32, logits_ref.shape, 1)
  # greedy_sampled = max_arrays(
  #   [logits_ref[...], token_idx], num_keys=1+int(stable), axis=1
  # )[1]
  # Reshape to (block_token, 1) to match output ref
  # greedy_sampled = jnp.expand_dims(greedy_sampled, axis=-1)

  # Top-k masking
  logits = topk_mask_ref_inputs(
    logits_ref, k_ref, 
    replace_val=replace_val,
    stable=stable,
    underlying_dtype=underlying_logits_dtype)
  logits /= temperature_ref[...].astype(logits.dtype)
  logits_max /= temperature_ref[...].astype(logits.dtype)

  # Top-p masking
  logits = topp_mask(
    logits, p_ref[...], replace_val=replace_val,
    return_unnorm_i32_probs=sample_in_i32,
    logits_max=logits_max
  )
  if not sample_in_i32:
    # Random key splitting is based on idx in ravelled array
    # We pass in (batch_idx, token_idx) for linearized position: batch_idx * vocab_size + token_idx
    batch_idx = lax.broadcasted_iota(jnp.int32, logits.shape, 0) + pl.program_id(0) * logits_ref.shape[0] + dim0_offset_ref[0]
    next_tokens = sparse_random_categorical(
      rng_key_ref,
      logits,
      (batch_idx, token_idx),
      dim1_size=logits.shape[1],
      axis=1,  # Sample along vocab axis
      dtype=jnp.float32,
    )[1]  # Take sampled token indices
  else:
    unnorm_probs_i32 = logits # alias
    # High-precision integer sampling
    total_sum_u48 = sum_in_u48(unnorm_probs_i32, scale_bits=24)
    # Split rng_key_ref (4, 2) into list of 4 keys for random_u48
    keys = [rng_key_ref[i][...] for i in range(4)]
    target_u48 = random_u48(keys, total_sum_u48, shape=(logits.shape[0], 1))
    next_tokens = find_boundary_idx(
      unnorm_probs_i32,
      map_fn=lambda x: U48(x, max_val=2**24-1),
      target=target_u48
    )
  # # Reshape to (block_token, 1) to match output ref
  if next_tokens.ndim == 1:
    next_tokens = jnp.expand_dims(next_tokens, axis=-1)

  sampled_tokens_ref[...] = jnp.where(temperature_ref[...] < _SAMPLING_EPS, greedy_sampled, next_tokens)


@functools.partial(
  jax.jit,
  static_argnames=["stable", "replace_val", "block_token", "interpret", "sample_in_i32"],
)
def topk_topp_mask_and_sample(
  logits: jax.Array,
  rng_key: jax.Array,
  k: jax.Array,
  p: jax.Array,
  temperature: jax.Array,
  dim0_offset: int = 0,
  *,
  stable: bool = True,
  replace_val: float = -1e12,
  block_token: int = 8,
  interpret: bool = False,
  sample_in_i32: bool = False,
) -> jax.Array:
  """Top-k, top-p masking and sampling using Pallas.

  Args:
    logits: Input logits [batch, vocab_size]
    rng_key: RNG key
    k: Top-k values [batch] or scalar
    p: Top-p values [batch] or scalar
    temperature: Temperature values [batch] or scalar
    dim0_offset: Offset for batch indexing
    stable: Whether to use stable masking
    replace_val: Replacement value for masked elements
    block_token: Number of tokens per block
    interpret: Whether to use interpret mode

  Returns:
    Sampled token indices [batch]
  """
  batch_size, vocab_size = logits.shape

  # Ensure inputs have correct shapes
  # First ensure they're at least 1D arrays, then reshape to (batch_size, 1)
  k = jnp.atleast_1d(k)
  p = jnp.atleast_1d(p)
  temperature = jnp.atleast_1d(temperature)

  # Broadcast to batch_size if scalar, then add dimension
  if k.shape[0] == 1:
    k = jnp.broadcast_to(k, (batch_size,))
  if p.shape[0] == 1:
    p = jnp.broadcast_to(p, (batch_size,))
  if temperature.shape[0] == 1:
    temperature = jnp.broadcast_to(temperature, (batch_size,))

  # Now reshape to (batch_size, 1)
  k = jnp.reshape(k, (batch_size, 1))
  p = jnp.reshape(p, (batch_size, 1))
  temperature = jnp.reshape(temperature, (batch_size, 1))
  dim0_offset_arr = jnp.array([dim0_offset], dtype=jnp.int32)

  # Prepare RNG key
  if not sample_in_i32:
    if rng_key.ndim == 0:
      rng_key = jax.random.key_data(rng_key)
    if rng_key.ndim == 1:
      rng_key = jnp.reshape(rng_key, (1, 2))
    elif rng_key.shape != (1, 2):
      # If it's already 2D, ensure it's (1, 2)
      rng_key = jnp.reshape(rng_key, (1, 2))
  else:
    # For U48 sampling, we need 4 keys
    rng_key = tuple(jax.random.split(rng_key, 4))

  # Pad batch to multiple of block_token
  num_blocks = pl.cdiv(batch_size, block_token)
  padded_batch = num_blocks * block_token

  if padded_batch != batch_size:
    pad_size = padded_batch - batch_size
    logits = jnp.pad(logits, ((0, pad_size), (0, 0)), constant_values=replace_val)
    k = jnp.pad(k, ((0, pad_size), (0, 0)), constant_values=1)
    p = jnp.pad(p, ((0, pad_size), (0, 0)), constant_values=1.0)
    temperature = jnp.pad(temperature, ((0, pad_size), (0, 0)), constant_values=1.0)

  output_shape = jax.ShapeDtypeStruct((padded_batch, 1), jnp.int32)

  result = pl.pallas_call(
    functools.partial(
      topk_topp_mask_and_sample_kernel,
      stable=stable,
      replace_val=replace_val,
      sample_in_i32=sample_in_i32,
    ),
    out_shape=output_shape,
    grid=(num_blocks,),
    in_specs=[
      pl.BlockSpec((block_token, vocab_size), lambda i: (i, 0)),  # logits
      (pl.BlockSpec(memory_space=pltpu.VMEM),)*4 if sample_in_i32 else pl.BlockSpec(memory_space=pltpu.SMEM),  # rng_key
      pl.BlockSpec((block_token, 1), lambda i: (i, 0)),  # k
      pl.BlockSpec((block_token, 1), lambda i: (i, 0)),  # p
      pl.BlockSpec((block_token, 1), lambda i: (i, 0)),  # temperature
      pl.BlockSpec(memory_space=pltpu.SMEM),  # dim0_offset
    ],
    out_specs=pl.BlockSpec((block_token, 1), lambda i: (i, 0)),
    compiler_params=pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 2**27)),
    interpret=interpret,
  )(logits, rng_key, k, p, temperature, dim0_offset_arr)

  # Remove padding
  result = result[:batch_size, 0]

  return result
