"""
Fused TPU sampling kernel implementing top-p filtering, temperature scaling,
and categorical sampling in a single Pallas kernel.
"""

import functools
import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tallax.tax.bitonic_topk import bitonic_topk_inner
from tallax.tax.sparse_random import sparse_random_uniform
from tallax.utils import NUM_LANES, NUM_SUBLANES, pad, to_32bit_dtype, log2
from jax import lax


def _fused_sampling_kernel(
    logits_ref,
    logits_global_index_ref,
    rng_key_ref,
    top_p_ref,
    temperature_ref,
    next_tokens_ref,
    greedy_sampled_ref,
):
    """
    Fused kernel implementing top-p filtering, temperature scaling, and sampling.

    Args:
        logits_ref: Input logits of shape (batch_size, vocab_size)
        logits_global_index_ref: Global vocabulary indices of shape (batch_size, vocab_size)
        rng_key_ref: RNG key for sampling of shape (1, 2)
        top_p_ref: Top-p threshold of shape (batch_size,)
        temperature_ref: Temperature values of shape (batch_size,)
        next_tokens_ref: Output sampled tokens of shape (batch_size,)
        greedy_sampled_ref: Output greedy tokens of shape (batch_size,)
    """
    batch_size, vocab_size = logits_ref.shape

    # Convert logits to float32
    logits = logits_ref[...].astype(jnp.float32)
    logits_global_index = logits_global_index_ref[...]

    # Step 1: Sort logits in descending order using bitonic_topk_inner
    # This gives us sorted logits and their corresponding global indices
    sorted_logits, sorted_indices = bitonic_topk_inner(
        [logits, logits_global_index],
        k=NUM_LANES,
        num_keys=1
    )

    # Step 2: Greedy sampling - just take the first sorted index (argmax)
    greedy_sampled = sorted_indices[:, 0]

    # Step 3: Compute softmax probabilities on sorted logits for top-p filtering
    # For numerical stability, subtract max (which is now the first element)
    logits_max = sorted_logits[:, 0:1]
    logits_shifted = sorted_logits - logits_max
    exp_logits = jnp.exp(logits_shifted)
    sorted_probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    # Step 4: Top-p filtering using cumsum on sorted probabilities
    # Implement cumsum using the same Hillis-Steele algorithm as tallax.tax.cumsum
    cumsum_probs = sorted_probs
    num_steps = log2(NUM_LANES)
    for step in range(num_steps):
        offset = 1 << step
        # Create indices using broadcasted_iota (TPU-friendly)
        idx = lax.broadcasted_iota(jnp.int32, (batch_size, NUM_LANES), 1)
        src_idx = lax.max(idx - offset, 0)

        # Gather shifted values
        shifted = jnp.take_along_axis(cumsum_probs, src_idx, axis=1)

        # Add shifted values where valid (index >= offset)
        valid_mask = idx >= offset
        cumsum_probs = jnp.where(valid_mask, cumsum_probs + shifted, cumsum_probs)

    # Mask sorted logits where cumulative probability > top_p
    top_p_expanded = jnp.expand_dims(top_p_ref[...], axis=-1)
    mask = cumsum_probs <= top_p_expanded
    sorted_logits_filtered = jnp.where(mask, sorted_logits, -1e12)

    # Step 5: Apply temperature scaling
    temperatures = temperature_ref[...].astype(sorted_logits.dtype)
    temperatures_expanded = jnp.expand_dims(temperatures, axis=-1)
    sorted_logits_scaled = sorted_logits_filtered / temperatures_expanded

    # Step 6: Categorical sampling using Gumbel-max trick
    # Generate Gumbel noise using sparse random uniform
    batch_indices = lax.broadcasted_iota(jnp.int32, (batch_size, NUM_LANES), 0)
    vocab_indices = lax.broadcasted_iota(jnp.int32, (batch_size, NUM_LANES), 1)

    u = sparse_random_uniform(
        rng_key_ref,
        (batch_indices, vocab_indices),
        NUM_LANES,
        dtype=jnp.float32,
        minval=jnp.finfo(jnp.float32).tiny,
        maxval=1.0
    )

    # Compute Gumbel noise: -log(-log(u))
    gumbel = -jnp.log(-jnp.log(u))

    # Add Gumbel noise to scaled logits
    logits_with_gumbel = sorted_logits_scaled + gumbel

    # Step 7: Find argmax using bitonic_topk_inner
    _, sampled_local_indices = bitonic_topk_inner(
        [logits_with_gumbel, lax.broadcasted_iota(jnp.int32, (batch_size, NUM_LANES), 1)],
        k=NUM_LANES,
        num_keys=1
    )

    # Map local index to global index
    # sampled_local_indices[:, 0] gives the position in sorted array
    # We need to get the global index from sorted_indices at that position
    next_tokens = jnp.take_along_axis(sorted_indices, sampled_local_indices[:, 0:1], axis=1).squeeze(1)

    # Write outputs
    next_tokens_ref[...] = next_tokens
    greedy_sampled_ref[...] = greedy_sampled


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
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """
    Fused TPU kernel for sampling with top-p filtering and temperature scaling.

    Implements the following operations in a single fused kernel:
    1. Convert logits to float32
    2. Apply top-p filtering using softmax and cumulative sum
    3. Apply temperature scaling
    4. Sample next tokens using categorical distribution (Gumbel-max trick)
    5. Compute greedy samples (argmax)

    Args:
        logits: Input logits of shape (batch_size, vocab_size)
        logits_global_index: Global vocabulary indices of shape (batch_size, vocab_size)
        rng_key: RNG key for sampling, shape (2,)
        top_p: Top-p threshold values, scalar or shape (batch_size,)
        temperature: Temperature values, scalar or shape (batch_size,)
        interpret: If True, run in CPU interpret mode (default: False)

    Returns:
        Tuple of:
        - next_tokens: Sampled tokens of shape (batch_size,)
        - greedy_sampled: Greedy tokens (argmax) of shape (batch_size,)

    Example:
        >>> batch_size, vocab_size = 8, 128
        >>> logits = jax.random.normal(jax.random.PRNGKey(0), (batch_size, vocab_size))
        >>> logits_global_index = jnp.tile(jnp.arange(vocab_size), (batch_size, 1))
        >>> rng_key = jax.random.PRNGKey(42)
        >>> top_p = jnp.full((batch_size,), 0.9)
        >>> temperature = jnp.full((batch_size,), 1.0)
        >>> next_tokens, greedy = fused_tpu_sampling(
        ...     logits, logits_global_index, rng_key, top_p, temperature
        ... )
    """
    batch_size, vocab_size = logits.shape

    # Pad logits to TPU-compatible shapes if needed
    # Ensure batch_size is padded to NUM_SUBLANES and vocab_size to NUM_LANES
    logits_padded = pad(logits, (NUM_SUBLANES, NUM_LANES), val='min')
    logits_global_index_padded = pad(
        logits_global_index, (NUM_SUBLANES, NUM_LANES), val=0
    )
    padded_batch_size, padded_vocab_size = logits_padded.shape

    # Broadcast scalar inputs to batch dimension
    top_p = jnp.broadcast_to(top_p, (padded_batch_size,))
    temperature = jnp.broadcast_to(temperature, (padded_batch_size,))

    # Reshape RNG key to (1, 2) for sparse_random_uniform
    rng_key_reshaped = rng_key.reshape(1, 2)

    # Define output shapes
    output_shapes = (
        jax.ShapeDtypeStruct((padded_batch_size,), jnp.int32),  # next_tokens
        jax.ShapeDtypeStruct((padded_batch_size,), jnp.int32),  # greedy_sampled
    )

    # Call the Pallas kernel with grid shape ()
    next_tokens, greedy_sampled = pl.pallas_call(
        _fused_sampling_kernel,
        out_shape=output_shapes,
        grid=(),  # Grid shape of () as requested
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(0.9 * 2**27)
        ),
        interpret=interpret,
    )(
        logits_padded,
        logits_global_index_padded,
        rng_key_reshaped,
        top_p,
        temperature,
    )

    # Unpad outputs
    return next_tokens[:batch_size], greedy_sampled[:batch_size]
