"""Tests for fused TPU sampling kernel."""

import jax
import jax.numpy as jnp
import pytest
from tallax.tax.fused_sampling import fused_tpu_sampling
from tallax.utils import NUM_LANES, NUM_SUBLANES


def test_fused_sampling_basic():
    """Test basic functionality of fused_tpu_sampling."""
    batch_size = NUM_SUBLANES
    vocab_size = NUM_LANES

    # Create sample inputs
    key = jax.random.PRNGKey(42)
    logits_key, rng_key = jax.random.split(key)

    logits = jax.random.normal(logits_key, (batch_size, vocab_size))
    logits_global_index = jnp.tile(jnp.arange(vocab_size, dtype=jnp.int32), (batch_size, 1))
    top_p = jnp.full((batch_size,), 0.9)
    temperature = jnp.full((batch_size,), 1.0)

    # Run sampling
    next_tokens, greedy_sampled = fused_tpu_sampling(
        logits, logits_global_index, rng_key, top_p, temperature, interpret=True
    )

    # Check output shapes
    assert next_tokens.shape == (batch_size,)
    assert greedy_sampled.shape == (batch_size,)

    # Check that outputs are valid vocabulary indices
    assert jnp.all(next_tokens >= 0)
    assert jnp.all(next_tokens < vocab_size)
    assert jnp.all(greedy_sampled >= 0)
    assert jnp.all(greedy_sampled < vocab_size)

    # Check that greedy sampling returns the argmax
    for i in range(batch_size):
        expected_greedy = jnp.argmax(logits[i])
        assert greedy_sampled[i] == expected_greedy, \
            f"Greedy sample mismatch at index {i}: expected {expected_greedy}, got {greedy_sampled[i]}"


def test_fused_sampling_temperature():
    """Test that temperature affects sampling."""
    batch_size = NUM_SUBLANES
    vocab_size = NUM_LANES

    key = jax.random.PRNGKey(123)
    logits_key, rng_key1, rng_key2 = jax.random.split(key, 3)

    # Create logits with one clear maximum
    logits = jax.random.normal(logits_key, (batch_size, vocab_size))
    logits = logits.at[:, 0].set(10.0)  # Make first token very likely

    logits_global_index = jnp.tile(jnp.arange(vocab_size, dtype=jnp.int32), (batch_size, 1))
    top_p = jnp.full((batch_size,), 1.0)  # No top-p filtering

    # Sample with low temperature (should be more deterministic)
    temperature_low = jnp.full((batch_size,), 0.1)
    next_tokens_low, _ = fused_tpu_sampling(
        logits, logits_global_index, rng_key1, top_p, temperature_low, interpret=True
    )

    # Sample with high temperature (should be more random)
    temperature_high = jnp.full((batch_size,), 2.0)
    next_tokens_high, _ = fused_tpu_sampling(
        logits, logits_global_index, rng_key2, top_p, temperature_high, interpret=True
    )

    # With low temperature, most samples should be 0 (the argmax)
    low_temp_zeros = jnp.sum(next_tokens_low == 0)
    high_temp_zeros = jnp.sum(next_tokens_high == 0)

    # Low temperature should produce more argmax samples
    assert low_temp_zeros >= high_temp_zeros, \
        f"Low temperature should be more deterministic: {low_temp_zeros} vs {high_temp_zeros}"


def test_fused_sampling_different_shapes():
    """Test with different input shapes."""
    for batch_size in [NUM_SUBLANES, NUM_SUBLANES * 2]:
        for vocab_size in [NUM_LANES, NUM_LANES * 2]:
            key = jax.random.PRNGKey(456)
            logits_key, rng_key = jax.random.split(key)

            logits = jax.random.normal(logits_key, (batch_size, vocab_size))
            logits_global_index = jnp.tile(
                jnp.arange(vocab_size, dtype=jnp.int32), (batch_size, 1)
            )
            top_p = jnp.full((batch_size,), 0.95)
            temperature = jnp.full((batch_size,), 1.0)

            next_tokens, greedy_sampled = fused_tpu_sampling(
                logits, logits_global_index, rng_key, top_p, temperature, interpret=True
            )

            assert next_tokens.shape == (batch_size,)
            assert greedy_sampled.shape == (batch_size,)


def test_fused_sampling_scalar_params():
    """Test with scalar top_p and temperature."""
    batch_size = NUM_SUBLANES
    vocab_size = NUM_LANES

    key = jax.random.PRNGKey(789)
    logits_key, rng_key = jax.random.split(key)

    logits = jax.random.normal(logits_key, (batch_size, vocab_size))
    logits_global_index = jnp.tile(jnp.arange(vocab_size, dtype=jnp.int32), (batch_size, 1))

    # Use scalar values
    top_p = 0.9
    temperature = 1.0

    next_tokens, greedy_sampled = fused_tpu_sampling(
        logits, logits_global_index, rng_key, top_p, temperature, interpret=True
    )

    assert next_tokens.shape == (batch_size,)
    assert greedy_sampled.shape == (batch_size,)


if __name__ == "__main__":
    test_fused_sampling_basic()
    test_fused_sampling_temperature()
    test_fused_sampling_different_shapes()
    test_fused_sampling_scalar_params()
    print("All tests passed!")
