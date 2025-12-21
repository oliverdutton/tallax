"""Test compressed transpose sorting mode."""

import pytest
import jax
import jax.numpy as jnp

from tallax._src.utils import is_cpu_platform
from tallax._src.sort import _sort_in_vmem, xla_equivalent_sort


def _should_skip_on_cpu(size):
    """Skip tests on CPU for large sizes (> 256) to avoid slow tests."""
    return is_cpu_platform() and size > 256


@pytest.mark.parametrize("size", [128, 2**10, 2**13])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_compressed_mode(size, descending):
    """Test compressed transpose mode in _sort_in_vmem."""
    if _should_skip_on_cpu(size):
        pytest.skip("Skipping large size on CPU")

    batch_size = 8
    shape = (batch_size, size)
    interpret = is_cpu_platform()

    # Generate random test data
    key = jax.random.key(0)
    x = jax.random.normal(key, shape, dtype=jnp.float32)

    # Sort using compressed mode
    sorted_x, = _sort_in_vmem(
        x,
        num_keys=1,
        descending=descending,
        interpret=interpret,
        use_compressed_throughout=True,
        unroll_compressed=128,
    )

    # Sort using reference
    expected, = xla_equivalent_sort(x, num_keys=1, descending=descending)

    assert jnp.allclose(sorted_x, expected), f"Mismatch for shape={shape}, descending={descending}"


@pytest.mark.parametrize("size", [128, 2**10])
def test_sort_compressed_vs_normal(size):
    """Compare compressed mode with normal mode."""
    if _should_skip_on_cpu(size):
        pytest.skip("Skipping large size on CPU")

    batch_size = 8
    shape = (batch_size, size)
    interpret = is_cpu_platform()

    key = jax.random.key(42)
    x = jax.random.normal(key, shape)

    # Sort using compressed mode
    sorted_compressed, = _sort_in_vmem(
        x, num_keys=1, interpret=interpret, use_compressed_throughout=True
    )

    # Sort using normal mode
    sorted_normal, = _sort_in_vmem(
        x, num_keys=1, interpret=interpret, use_compressed_throughout=False
    )

    assert jnp.allclose(sorted_compressed, sorted_normal), \
        "Compressed and normal modes should produce same results"
