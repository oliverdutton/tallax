"""Tallax sampling module.

Public API for TPU-optimized sampling operations.
"""

import functools
import jax
import jax.numpy as jnp
from jax import jit

# Import from _src module using JAX's pattern
# The "as <name>" syntax is required for proper re-export
from tallax._src.sampling import top_p_mask as top_p_mask
from tallax._src.sampling import top_p_and_sample_arrays as top_p_and_sample_arrays
from tallax._src.sampling import top_p_and_sample_refs as top_p_and_sample_refs
from tallax._src.sampling import top_p_and_sample as top_p_and_sample
from tallax._src.sampling import sample as sample
from tallax._src.sampling import _top_k_with_sharding
from tallax._src.utils import NUM_LANES


@functools.partial(
    jit,
    static_argnames=("interpret",),
)
def fused_tpu_sampling(
    logits: jax.Array,
    logits_global_index: jax.Array,
    rng_key: jax.Array,
    top_p: jax.Array,
    temperature: jax.Array,
    *,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """
    Fused TPU sampling with top-p filtering and temperature scaling.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        logits_global_index: Global indices for logits of shape (batch_size, vocab_size)
        rng_key: RNG key for sampling
        top_p: Top-p threshold(s), scalar or shape (batch_size,)
        temperature: Temperature value(s), scalar or shape (batch_size,)
        interpret: If True, run in CPU interpret mode (default: False)

    Returns:
        Tuple of (next_tokens, greedy_sampled):
            - next_tokens: Sampled tokens of shape (batch_size,)
            - greedy_sampled: Greedy (argmax) tokens of shape (batch_size,)
    """
    batch_size, vocab_size = logits.shape

    # Ensure top_p and temperature are arrays
    if jnp.ndim(top_p) == 0:
        top_p = jnp.full((batch_size,), top_p)
    if jnp.ndim(temperature) == 0:
        temperature = jnp.full((batch_size,), temperature)

    # Get top-k for all samples (using NUM_LANES as k)
    k = NUM_LANES
    topk_logits, topk_idx = _top_k_with_sharding(
        logits,
        k=jnp.full((batch_size,), k, dtype=jnp.int32),
        replace_val=-1e12
    )

    # Get greedy samples (argmax)
    greedy_sampled = topk_idx[:, 0]

    # Sample with top-p and temperature
    next_tokens = top_p_and_sample(
        topk_logits,
        topk_idx,
        rng_key,
        top_p=top_p,
        temperature=temperature,
        vocab_size=vocab_size,
        replace_val=-1e12,
        interpret=interpret,
    )

    return next_tokens, greedy_sampled


__all__ = [
    "top_p_mask",
    "top_p_and_sample_arrays",
    "top_p_and_sample_refs",
    "top_p_and_sample",
    "sample",
    "fused_tpu_sampling",
]
