"""Test BoundedInt with bitonic sort on (16, 1024) shape."""

import time
import jax
import jax.numpy as jnp
from tallax._src.utils import is_cpu_platform
from tallax._src.bitonic_sort import bitonic_sort


def simple_benchmark(fn, name="Function", warmup=2, runs=5):
    """Simple benchmark function."""
    print(f"\nBenchmarking {name}...")

    # Warmup
    for _ in range(warmup):
        _ = fn()

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"  Average: {avg_time*1000:.2f}ms")
    print(f"  Min:     {min_time*1000:.2f}ms")
    print(f"  Max:     {max_time*1000:.2f}ms")

    return avg_time


def test_bitonic_sort_with_bounded_int():
    """Test bitonic sort with BoundedInt on (16, 1024) shape."""

    shape = (16, 1024)
    interpret = is_cpu_platform()

    print("=" * 70)
    print(f"Testing BoundedInt with bitonic sort on shape {shape}")
    print("Configuration: stage_unroll=6, tile_unroll=None")
    print("=" * 70)

    # Create test data
    key = jax.random.key(0)
    logits = jax.random.normal(key, shape, dtype=jnp.float32).astype(jnp.bfloat16)

    print(f"\nInput shape: {logits.shape}")
    print(f"Input dtype: {logits.dtype}")
    print(f"First row (first 10 elements): {logits[0, :10]}")

    # Test configuration
    kwargs = {
        'stage_unroll': 6,
        'tile_unroll': None,
    }

    print(f"\nTest parameters: {kwargs}")

    # Run bitonic sort
    print("\n" + "-" * 70)
    print("Compiling and running bitonic_sort...")
    print("-" * 70)

    result = bitonic_sort(logits, **kwargs, interpret=interpret)

    print(f"\nOutput shape: {result[0].shape}")
    print(f"Output dtype: {result[0].dtype}")
    print(f"First row (first 10 elements): {result[0][0, :10]}")

    # Verify correctness against JAX's built-in sort
    print("\n" + "-" * 70)
    print("Verifying correctness...")
    print("-" * 70)

    expected = jax.lax.sort(logits, dimension=1)
    matches = (result[0] == expected).mean()

    print(f"Match rate with jax.lax.sort: {matches * 100:.2f}%")

    # Check if sorted correctly
    is_sorted = jnp.all(result[0][:, :-1] <= result[0][:, 1:])
    print(f"Output is sorted: {is_sorted}")

    # Show a few sample values to verify sorting
    print(f"\nSample verification (row 0, indices 0, 500, 1000, 1023):")
    print(f"  Result:   [{float(result[0][0, 0]):.4f}, {float(result[0][0, 500]):.4f}, {float(result[0][0, 1000]):.4f}, {float(result[0][0, 1023]):.4f}]")
    print(f"  Expected: [{float(expected[0, 0]):.4f}, {float(expected[0, 500]):.4f}, {float(expected[0, 1000]):.4f}, {float(expected[0, 1023]):.4f}]")

    # Benchmark
    print("\n" + "-" * 70)
    print("Running benchmark...")
    print("-" * 70)

    bitonic_time = simple_benchmark(
        lambda: bitonic_sort(logits, **kwargs, interpret=interpret),
        name="bitonic_sort with BoundedInt"
    )

    jax_time = simple_benchmark(
        lambda: jax.lax.sort(logits),
        name="jax.lax.sort"
    )

    print(f"\nSpeedup: {jax_time / bitonic_time:.2f}x")

    return result, expected, matches


def test_with_different_dtypes():
    """Test with different data types."""

    shape = (16, 1024)
    interpret = is_cpu_platform()
    kwargs = {
        'stage_unroll': 6,
        'tile_unroll': None,
    }

    print("\n" + "=" * 70)
    print("Testing with different dtypes")
    print("=" * 70)

    for dtype in [jnp.float32, jnp.bfloat16, jnp.int32]:
        print(f"\nTesting with dtype: {dtype}")

        key = jax.random.key(42)
        if jnp.issubdtype(dtype, jnp.integer):
            logits = jax.random.randint(key, shape, -1000, 1000, dtype=dtype)
        else:
            logits = jax.random.normal(key, shape, dtype=jnp.float32).astype(dtype)

        result = bitonic_sort(logits, **kwargs, interpret=interpret)
        expected = jax.lax.sort(logits, dimension=1)

        matches = (result[0] == expected).mean()
        is_sorted = jnp.all(result[0][:, :-1] <= result[0][:, 1:])

        print(f"  Match rate: {matches * 100:.2f}%")
        print(f"  Is sorted: {is_sorted}")


def test_descending_order():
    """Test descending sort."""

    shape = (16, 1024)
    interpret = is_cpu_platform()
    kwargs = {
        'stage_unroll': 6,
        'tile_unroll': None,
        'descending': True
    }

    print("\n" + "=" * 70)
    print("Testing descending sort")
    print("=" * 70)

    key = jax.random.key(123)
    logits = jax.random.normal(key, shape, dtype=jnp.float32).astype(jnp.bfloat16)

    result = bitonic_sort(logits, **kwargs, interpret=interpret)

    is_descending = jnp.all(result[0][:, :-1] >= result[0][:, 1:])
    print(f"Output is sorted in descending order: {is_descending}")
    print(f"First row (first 10 elements): {result[0][0, :10]}")
    print(f"First row (last 10 elements):  {result[0][0, -10:]}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BoundedInt Bitonic Sort Test Suite")
    print("=" * 70)

    # Main test
    result, expected, matches = test_bitonic_sort_with_bounded_int()

    # Additional tests
    test_with_different_dtypes()
    test_descending_order()

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)

    if matches > 0.99:
        print("\n✓ SUCCESS: BoundedInt integration working correctly!")
    else:
        print(f"\n✗ WARNING: Match rate is {matches * 100:.2f}%, expected >99%")
