
import pytest
import jax
import jax.numpy as jnp

from tallax.utils import is_cpu_platform
from tallax.test_utils import verify_sort_output


@pytest.mark.skipif(
    is_cpu_platform(),
    reason="Sort tests require TPU/GPU - CPU uses interpret mode which is slow for comprehensive tests"
)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("size", [128, 2048, 131072])
@pytest.mark.parametrize("variant", [
    "standard",
    "return_argsort",
    "return_argsort_stable",
    "descending",
    "descending_argsort",
    "descending_stable"
])
@pytest.mark.parametrize("num_arrays,num_keys", [
    (1, 1),
    (2, 1),
    (2, 2)
])
def test_sort_comprehensive(dtype, size, variant, num_arrays, num_keys):
    """Comprehensive sort tests with various configurations."""
    shape = (16, size)
    key = jax.random.key(0)

    # Generate operands
    operands = []
    for i in range(num_arrays):
        if dtype == jnp.bfloat16:
            arr = jax.random.normal(jax.random.fold_in(key, i), shape, dtype=jnp.float32).astype(jnp.bfloat16)
        else:
            arr = jax.random.normal(jax.random.fold_in(key, i), shape, dtype=dtype)
        operands.append(arr)

    # Parse variant
    return_argsort = "return_argsort" in variant
    is_stable = "stable" in variant
    descending = "descending" in variant

    verify_sort_output(
        operands,
        num_keys=num_keys,
        return_argsort=return_argsort,
        is_stable=is_stable,
        descending=descending,
        interpret=False
    )
