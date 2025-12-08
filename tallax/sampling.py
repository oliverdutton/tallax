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

from tallax.tax.bitonic_topk import bitonic_topk_inner as topk, top1
from tallax.tax.gather import pallas_compatible_take_along_axis as take_along_axis
from tallax.tax.sparse_random import sparse_random_uniform
from tallax.tax.cumsum import pallas_compatible_cumsum as cumsum
from tallax.utils import NUM_LANES, NUM_SUBLANES, pad, log2, iota_tile, transpose_list_of_lists

_SAMPLING_EPS = 1e-5


def topp_mask(*, topk_logits, p, replace_val, axis):
    """
    Apply top-p filtering mask to sorted logits.

    Args:
        topk_logits: Sorted logits (descending order)
        p: Top-p threshold(s)
        replace_val: Value to replace filtered logits with
        axis: Axis along which to apply filtering (must be 0)

    Returns:
        Masked logits with values outside top-p set to replace_val
    """
    if axis != 0:
        raise NotImplementedError("topp_mask only supports axis=0")

    shape = topk_logits.shape

    # Compute softmax probabilities
    # For numerical stability, subtract max (pre-sorted so its the first element)
    exp_logits = jnp.exp(topk_logits - topk_logits[:1,:])
    probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)

    # Top-p filtering using cumsum on sorted probabilities
    cumsum_probs = cumsum(probs, axis=0)

    # Find last idx where top-p probability mass is not covered
    threshold_idx = (cumsum_probs < p[None,:]).sum(0, keepdims=True)
    # vLLM current implementation uses binary search, computing a threshold.
    # so ties at the threshold are all included
    # we replicate that behavior here
    thresholds = take_along_axis(
        topk_logits, jnp.broadcast_to(threshold_idx, shape), 0)
    topp_logits = jnp.where(
        topk_logits >= thresholds,
        topk_logits, replace_val)

    return topp_logits


def top_p_and_sample_jax_inner(*, topk_logits, topk_idx, rng_key, top_p, temperature, vocab_size, replace_val):
    """
    Implements top-p filtering, temperature scaling, and sampling.
    """
    # Convert logits to float32
    topk_logits = topk_logits.astype(jnp.float32)

    # To do reductions and broadcast across sublanes rather than lanes (which are slow)
    # we shift sampling to dim 0
    topk_logits = topk_logits.T
    topk_idx = topk_idx.T
    shape = topk_logits.shape

    # Apply top-p masking
    topp_logits = topp_mask(
        topk_logits=topk_logits,
        p=top_p,
        replace_val=replace_val,
        axis=0
    )

    # Step 5: Apply temperature scaling
    topp_logits_scaled = topp_logits / temperature[None,:].astype(topp_logits.dtype)

    # Step 6: Categorical sampling using Gumbel-max trick
    # Generate Gumbel noise using sparse random uniform
    # random key splitting is based on idx in  ravelled array
    # we pass in (batch_idx.T, token_idx.T) and sample across axis 0, taking the token_idx
    batch_idx = lax.broadcasted_iota(jnp.int32, shape, 1)
    next_tokens = sparse_random_categorical(
        rng_key,
        topp_logits_scaled,
        # these are both transposed, (token, batch) shape
        (batch_idx, topk_idx),
        dim1_size=vocab_size,
        axis=0,
        dtype=jnp.float32
        # take token_idx
    )[1].squeeze(0)

    greedy_sampled = topk_idx[0,:]
    return jnp.where(
      temperature < _SAMPLING_EPS,
      greedy_sampled, next_tokens)


def top_p_and_sample_kernel(
    topk_logits_ref,
    topk_idx_ref,
    rng_key_ref,
    top_p_ref,
    temperature_ref,
    sampled_tokens_ref,
    *,
    vocab_size: int,
    replace_val: float,
):
    """
    Fused kernel implementing top-p filtering, temperature scaling, and sampling.
    """
    sampled_tokens_ref[...] = top_p_and_sample_jax_inner(
      topk_logits=topk_logits_ref[...],
      topk_idx=topk_idx_ref[...],
      rng_key=rng_key_ref, # SMEM, so keep as ref
      top_p=top_p_ref[...],
      temperature=temperature_ref[...],
      vocab_size=vocab_size,
      replace_val=replace_val,
    )

@functools.partial(
    jit,
    static_argnames=("vocab_size", "replace_val", "interpret",),
)
def top_p_and_sample(
    topk_logits: jax.Array,
    topk_idx: jax.Array,
    rng_key: jax.Array, # threefry2x32 key
    top_p: jax.Array,
    temperature: jax.Array,
    *,
    vocab_size: int,
    replace_val: float,
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
          top_p_and_sample_kernel, 
          vocab_size=vocab_size,
          replace_val=replace_val
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