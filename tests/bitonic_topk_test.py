import jax
import jax.numpy as jnp
import pytest

from tallax import tax
from tallax.utils import is_cpu_platform
from tallax.test_utils import verify_topk_output


# Test configurations
TEST_SHAPES = [
    (t, v)
    for t in [8, 32, 128]
    for v in [128, 1024, 2048]
    if t <= 128
]

@pytest.mark.skipif(
    is_cpu_platform(),
    reason="bitonic_topk requires TPU - CPU uses interpret mode which is slow"
)
@pytest.mark.parametrize("num_tokens,vocab_size", TEST_SHAPES)
def test_bitonic_topk(num_tokens, vocab_size):
    """Test bitonic_topk Pallas implementation."""
    k = 128

    # Generate test data
    key = jax.random.key(42)
    total_size = num_tokens * vocab_size
    # Use permutation to ensure unique values for strict equality check
    x = -jax.random.permutation(key, total_size).reshape(num_tokens, vocab_size).astype(jnp.int32)
    indices = jax.lax.broadcasted_iota(jnp.int32, (num_tokens, vocab_size), 1)

    # Run Pallas implementation
    # We pass (x, indices) to get both back sorted by x
    result = tax.bitonic_topk(
        (x, indices),
        k=k,
        num_keys=1,
        descending=True,
        interpret=False
    )

    # Validate results using verify_topk_output
    validation = verify_topk_output(x, result)

    assert bool(validation.all()), (
        f"bitonic_topk validation failed for shape {(num_tokens, vocab_size)}: "
        f"{int(validation.sum())}/{num_tokens} rows passed"
    )
