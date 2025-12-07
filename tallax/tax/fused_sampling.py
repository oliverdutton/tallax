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
from tallax.tax.sparse_random import sparse_random_uniform
from tallax.tax.cumsum import pallas_compatible_cumsum as cumsum
from tallax.utils import NUM_LANES, NUM_SUBLANES, pad, log2, iota_tile, transpose_list_of_lists

_SAMPLING_EPS = 1e-5


def top_p_and_sample_jax_inner(*, topk_logits, topk_idx, rng_key, top_p, temperature, vocab_size, replace_val):
    """
    Implements top-p filtering, temperature scaling, and sampling.
    """
    shape = topk_logits.shape

    # Convert logits to float32
    topk_logits = topk_logits.astype(jnp.float32)
    
    topk_logits = topk_logits.T
    topk_idx = topk_idx.T
    shape = shape[::-1]

    # Step 3: jax.nn.softmax
    # For numerical stability, subtract max (pre-sorted so its the first element)
    exp_logits = jnp.exp(topk_logits - topk_logits[:1,:])
    probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)

    # Step 4: Top-p filtering using cumsum on sorted probabilities
    # do in axis 0 as its faster, avoids some lane permutes
    cumsum_probs = cumsum(probs, axis=0)

    # Find last idx where top-p probability mass is not covered
    threshold_idx = (cumsum_probs < top_p[None,:]).sum(0, keepdims=True)
    # vLLM current implementation uses binary search, computing a threshold.
    # so ties at the threshold are all included    
    # we replicate that behavior here
    thresholds = jnp.take_along_axis(
      topk_logits, jnp.broadcast_to(threshold_idx, shape, 0)
    topp_logits = jnp.where(
    #jax.lax.broadcasted_iota(jnp.int32, shape, 1) < threshold_idx + 1,
    topk_logits >= thresholds,
    topk_logits, replace_val)

    # Step 5: Apply temperature scaling
    topp_logits_scaled = topp_logits / temperature[None,:].astype(topp_logits.dtype)

    # Step 6: Categorical sampling using Gumbel-max trick
    # Generate Gumbel noise using sparse random uniform
    dim0_idx = lax.broadcasted_iota(jnp.int32, shape, 1)
    u = sparse_random_uniform(
        rng_key,
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
    sampled_tokens = top1(
        [gumbel_logits, topk_idx],
        num_keys=1,
        axis=0
    )[1].squeeze(0)
    return jnp.where(
      temperature < _SAMPLING_EPS,
      topk_idx[0,:], sampled_tokens)

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