#!/usr/bin/env python3
"""
Test and benchmark script for bitonic_topk implementation.

Tests the Pallas-based bitonic_topk against JAX XLA top_k reference
for various input shapes with int32 values, with and without indices.
"""

import jax
import jax.numpy as jnp
import pytest

from tallax.tax.bitonic_topk import bitonic_topk
from tallax.utils import is_cpu_platform
from tallax.test_utils import verify_topk_output, benchmark


# Test configurations
NUM_TOKENS = [8, 16, 32, 64, 128]
VOCAB_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]

# Generate test cases: (num_tokens, vocab_size)
TEST_SHAPES = [(t, v) for t in NUM_TOKENS for v in VOCAB_SIZES if t <= 128]


@pytest.mark.skipif(
    is_cpu_platform(),
    reason="bitonic_topk requires TPU - CPU uses interpret mode which is slow"
)
@pytest.mark.parametrize("num_tokens,vocab_size", TEST_SHAPES)
def test_bitonic_topk_values_only(num_tokens, vocab_size):
    """Test bitonic_topk with values only (int32, descending)."""
    k = 128
    shape = (num_tokens, vocab_size)

    # Generate test data - negative random permutation
    total_size = num_tokens * vocab_size
    key = jax.random.PRNGKey(42)
    x = -jax.random.permutation(key, total_size).reshape(shape).astype(jnp.int32)

    # Run Pallas bitonic_topk
    result = bitonic_topk(x, k=k, descending=True, interpret=False)

    # Generate XLA reference
    xla_result = jax.vmap(lambda y: jax.lax.top_k(y, k))(x)

    # Extract values
    pallas_values = result[0] if isinstance(result, tuple) else result
    xla_values = xla_result[0]

    # Validate: check if sorted values match (handles ties)
    pallas_sorted = jnp.sort(pallas_values, axis=-1)[:, ::-1]
    xla_sorted = jnp.sort(xla_values, axis=-1)[:, ::-1]

    assert jnp.allclose(pallas_sorted, xla_sorted), (
        f"bitonic_topk failed for shape {shape}\n"
        f"First row Pallas: {pallas_values[0, :10]}\n"
        f"First row XLA:    {xla_values[0, :10]}"
    )


@pytest.mark.skipif(
    is_cpu_platform(),
    reason="bitonic_topk requires TPU - CPU uses interpret mode which is slow"
)
@pytest.mark.parametrize("num_tokens,vocab_size", TEST_SHAPES)
def test_bitonic_topk_with_indices(num_tokens, vocab_size):
    """Test bitonic_topk with indices (int32, descending)."""
    k = 128
    shape = (num_tokens, vocab_size)

    # Generate test data
    total_size = num_tokens * vocab_size
    key = jax.random.PRNGKey(42)
    x = -jax.random.permutation(key, total_size).reshape(shape).astype(jnp.int32)
    indices = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    # Run Pallas bitonic_topk with indices
    result = bitonic_topk((x, indices), k=k, num_keys=1, descending=True, interpret=False)

    # Validate using verify_topk_output
    # Note: verify_topk_output expects descending=True order
    validation = verify_topk_output(x, result)

    assert bool(validation.all()), (
        f"bitonic_topk with indices failed for shape {shape}: "
        f"{int(validation.sum())}/{num_tokens} rows passed validation"
    )


@pytest.mark.skipif(
    is_cpu_platform(),
    reason="bitonic_topk requires TPU - CPU uses interpret mode which is slow"
)
def test_bitonic_topk_iota():
    """Special test case: iota pattern."""
    k = 128
    shape = (8, 128)

    iota_data = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    # Pallas result
    pallas_result = bitonic_topk(iota_data, k=k, descending=True, interpret=False)

    # XLA reference
    xla_result = jax.vmap(lambda y: jax.lax.top_k(y, k))(iota_data)

    # Extract values
    pallas_values = pallas_result[0] if isinstance(pallas_result, tuple) else pallas_result
    xla_values = xla_result[0]

    assert jnp.allclose(pallas_values, xla_values), (
        f"bitonic_topk iota test failed\n"
        f"Expected: {xla_values[0]}\n"
        f"Got:      {pallas_values[0]}"
    )


def benchmark_bitonic_topk():
    """Benchmark bitonic_topk performance."""
    print("\n" + "="*80)
    print("BITONIC TOP-K BENCHMARK")
    print("="*80)

    test_cases = [
        (8, 2048),
        (8, 4096),
        (16, 2048),
        (32, 2048),
        (64, 2048),
        (128, 2048),
    ]

    for num_tokens, vocab_size in test_cases:
        shape = (num_tokens, vocab_size)
        print(f"\nShape: {shape}")

        # Generate test data
        total_size = num_tokens * vocab_size
        key = jax.random.PRNGKey(42)
        x = -jax.random.permutation(key, total_size).reshape(shape).astype(jnp.int32)

        # Benchmark Pallas
        print("Pallas bitonic_topk:")
        benchmark(lambda: bitonic_topk(x, k=128, descending=True, interpret=False))

        # Benchmark XLA
        print("XLA top_k:")
        benchmark(lambda: jax.vmap(lambda y: jax.lax.top_k(y, 128))(x))

        print("-" * 40)


if __name__ == "__main__":
    # Run benchmarks when executed directly
    print("Running benchmarks...")
    benchmark_bitonic_topk()
