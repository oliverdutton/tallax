"""Tests for the compressed transpose format sort implementation."""

import pytest
import jax
import jax.numpy as jnp

from tallax._src.utils import is_cpu_platform
from tallax._src.sort_compressed import sort_compressed
from tallax._src.sort import xla_equivalent_sort


def _should_skip_on_cpu(size):
    """Skip tests on CPU for large sizes (> 256) to avoid slow tests."""
    return is_cpu_platform() and size > 256


@pytest.mark.parametrize("dtype", [jnp.float32])
@pytest.mark.parametrize("size", [128, 256, 2**10, 2**13])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_compressed_basic(dtype, size, descending):
    """Test compressed format sort against XLA reference."""
    # Skip large sizes on CPU
    if _should_skip_on_cpu(size):
        pytest.skip("Skipping large size on CPU - interpret mode is too slow")

    batch_size = 8
    shape = (batch_size, size)

    # Use interpret mode on CPU
    interpret = is_cpu_platform()

    # Generate random test data
    key = jax.random.key(0)
    x = jax.random.normal(key, shape, dtype=jnp.float32).astype(dtype)

    # Sort using compressed format implementation
    sorted_x, = sort_compressed(x, num_keys=1, descending=descending, interpret=interpret)

    # Sort using reference implementation
    expected, = xla_equivalent_sort(x, num_keys=1, descending=descending)

    # Verify exact match
    assert jnp.allclose(sorted_x, expected), f"Mismatch for shape={shape}, descending={descending}"


@pytest.mark.parametrize("size", [128, 2**10])
def test_sort_compressed_reverse_sorted_input(size):
    """Test with reverse-sorted input (worst case for bitonic sort)."""
    if _should_skip_on_cpu(size):
        pytest.skip("Skipping large size on CPU")

    batch_size = 8
    shape = (batch_size, size)
    interpret = is_cpu_platform()

    # Create reverse-sorted input
    x = jnp.arange(batch_size * size, dtype=jnp.float32).reshape(shape)[:, ::-1]

    # Sort using compressed format
    sorted_x, = sort_compressed(x, num_keys=1, descending=False, interpret=interpret)

    # Sort using reference
    expected, = xla_equivalent_sort(x, num_keys=1, descending=False)

    assert jnp.allclose(sorted_x, expected)


@pytest.mark.parametrize("size", [128, 256])
@pytest.mark.parametrize("num_keys", [1, 2])
def test_sort_compressed_multi_key(size, num_keys):
    """Test multi-key sorting."""
    batch_size = 8
    shape = (batch_size, size)
    interpret = is_cpu_platform()

    # Generate multiple operands
    key = jax.random.key(42)
    keys = jax.random.split(key, 2)
    operands = [jax.random.normal(k, shape) for k in keys]

    # Sort using compressed format
    sorted_operands = sort_compressed(operands, num_keys=num_keys, interpret=interpret)

    # Sort using reference
    expected = xla_equivalent_sort(operands, num_keys=num_keys)

    for got, exp in zip(sorted_operands, expected):
        assert jnp.allclose(got, exp), f"Mismatch for num_keys={num_keys}"
