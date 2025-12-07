"""
Fused TPU sampling kernel implementing top-p filtering, temperature scaling,
and categorical sampling in a single Pallas kernel.
"""

import functools
import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.tax.bitonic_topk import bitonic_topk_inner
from tallax.tax.sparse_random import sparse_random_uniform
from tallax.utils import NUM_LANES, NUM_SUBLANES, pad, to_32bit_dtype, log2

_SAMPLING_EPS = 1e-5 

def cumsum_tile(tile, axis):
  n = tile.shape[axis]
  for stage in range(log2(n)):
    permutation = jax.lax.broadcasted_iota(jnp.int32, tile.shape, axis) - 2**stage
    tile += jnp.where(
      permutation>=0,
      jnp.take_along_axis(tile, permutation % n, axis=axis),
      axis)
  return tile

def cumsum(arr, axis):
  assert arr.ndim==2
  assert axis == 0
  shape = arr.shape
  arr = pad(arr, (NUM_SUBLANES, NUM_LANES))
  
  def _cumsum(arr):
    tiles = [arr[i*NUM_SUBLANES:(i+1)*NUM_SUBLANES] for i in range(arr.shape[0] // NUM_SUBLANES)]
    n = len(tiles)
    outs = [cumsum_tile(tile, axis) for tile in tiles]
    tile_sums = [tile.sum(axis, keepdims=True) for tile in tiles]
    for i in range(1, n): 
      outs[i] += tile_sums[i-1]
      tile_sums[i] += tile_sums[i-1]
    return jnp.concatenate(outs, axis=0)
    
  return jnp.concatenate(
    [
      _cumsum(arr[:,i*NUM_LANES:(i+1)*NUM_LANES]) for i in range(arr.shape[1] // NUM_LANES)
    ], axis=1
  )[:shape[0], :shape[1]]


def fused_sampling_kernel(
    topk_logits_ref,
    topk_idx_ref,
    rng_key_ref,
    top_p_ref,
    temperature_ref,
    sampled_tokens_ref,
    *,
    vocab_size: int,
):
    """
    Fused kernel implementing top-p filtering, temperature scaling, and sampling.
    """
    shape = topk_logits_ref.shape

    # Convert logits to float32
    topk_logits = topk_logits_ref[...].astype(jnp.float32)
    topk_idx = topk_idx_ref[...]

    # Step 3: jax.nn.softmax
    # For numerical stability, subtract max (pre-sorted so its the first element)
    exp_logits = jnp.exp(topk_logits - topk_logits[:,:1])
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    # Step 4: Top-p filtering using cumsum on sorted probabilities
    # do in dim0 as its faster, avoids some lane permutes
    cumsum_probs = cumsum(probs.T, axis=0).T

    # Find minimal number of values to cumsum >= p
    # count how many values cumsum don't cover top-p, then add 1 to find the threshold
    threshold_idx = (cumsum_probs < top_p_ref[...][:, None]).sum(1, keepdims=True)
    # vLLM current implementation uses binary search, computing a threshold. this includes ties in value at the top-p boundary for sampling. we replicate that behavior here
    thresholds = jnp.take_along_axis(
      topk_logits, jnp.broadcast t_to(threshold_idx, shape), 1)
    # we must cover the mass
    topp_logits = jnp.where(
    #jax.lax.broadcasted_iota(jnp.int32, shape, 1) < num_topp_vals,
    topk_logits >= thresholds,
    topk_logits, -1e12)

    # Step 5: Apply temperature scaling
    topp_logits_scaled = topp_logits / temperature_ref[...][:, None].astype(topp_logits.dtype)

    # Step 6: Categorical sampling using Gumbel-max trick
    # Generate Gumbel noise using sparse random uniform
    dim0_idx = lax.broadcasted_iota(jnp.int32, topk_idx.shape, 0)
    u = sparse_random_uniform(
        rng_key_ref,
        (dim0_idx, topk_idx),
        dim1_size=vocab_size,
        dtype=jnp.float32,
        minval=jnp.finfo(jnp.float32).tiny,
        maxval=1.0
    )
    # Compute Gumbel noise: -log(-log(u))
    gumbel = -jnp.log(-jnp.log(u))
    # Add Gumbel noise to scaled logits
    gumbel_logits = topp_logits_scaled + gumbel
    # Find argmax of Gumbel-perturbed logits
    # Since we only need the argmax (k=1), use bitonic_topk_inner with k=1
    sampled_tokens = bitonic_topk_inner(
        [gumbel_logits, topk_idx],
        k=1,
        num_keys=1
    )[1].squeeze(1)
        
    sampled_tokens_ref[...] = jnp.where(
      temperature_ref[...] < _SAMPLING_EPS,
      topk_idx[:, 0], sampled_tokens)

@functools.partial(
    jit,
    static_argnames=("vocab_size", "interpret",),
)
def fused_tpu_sampling(
    topk_logits: jax.Array,
    topk_idx: jax.Array,
    rng_key: jax.Array, # threefry2x32 key
    top_p: jax.Array,
    temperature: jax.Array,
    *,
    vocab_size: int,
    interpret: bool = False,
) -> jax.Array:
    """
    Fused TPU kernel for sampling with top-p filtering and temperature scaling.
    Padding logic has been removed.

    Args:
        topk_logits: Sorted logits of shape (batch_size, k)
        topk_idx: Indices corresponding to sorted logits of shape (batch_size, k)
        rng_key: RNG key for sampling, shape (2,)
        top_p: Top-p threshold values, scalar or shape (batch_size,)
        temperature: Temperature values, scalar or shape (batch_size,)
        interpret: If True, run in CPU interpret mode (default: False)

    Returns:
        next_tokens: Sampled tokens of shape (batch_size,)
    """
    return pl.pallas_call(
        functools.partial(
          fused_sampling_kernel, 
          vocab_size=vocab_size,
        ),
        in_specs=(
          pl.BlockSpec(),
          pl.BlockSpec(),
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(),
          pl.BlockSpec(),      
        ),
        out_shape=jax.ShapeDtypeStruct(topk_logits.shape[:1], jnp.int32),
        interpret=interpret,
    )(
        topk_logits,
        topk_idx,
        rng_key.reshape(1,2),
        top_p,
        temperature,
    )