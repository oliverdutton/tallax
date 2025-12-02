import pytest
import jax
import jax.numpy as jnp

from tallax.tax.bitonic_top_k import bitonic_top_k
from tallax.utils import is_cpu_platform, NUM_LANES
from tallax.test_utils import verify_topk_output


@pytest.mark.skipif(
    is_cpu_platform(),
    reason="bitonic_top_k requires TPU - CPU uses interpret mode which is slow"
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
@pytest.mark.parametrize("num_tokens", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("vocab_size", [128, 256, 1024, 2048])
@pytest.mark.parametrize("with_indices", [True, False])
def test_bitonic_top_k_comprehensive(dtype, num_tokens, vocab_size, with_indices):
    """Comprehensive bitonic_top_k tests with various configurations."""
    k = 128
    shape = (num_tokens, vocab_size)
    key = jax.random.key(0)

    # Generate random data
    if jnp.issubdtype(dtype, jnp.floating):
        x = jax.random.uniform(key, shape, dtype=dtype)
    else:
        x = jax.random.randint(key, shape, minval=-(2**30), maxval=2**30, dtype=dtype)

    operands = [x]
    if with_indices:
        indices = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
        operands.append(indices)

    operands = tuple(operands)

    # Run Pallas bitonic_top_k
    # We always use descending=True for top_k
    result = bitonic_top_k(operands, k=k, num_keys=1, descending=True, interpret=False)

    if with_indices:
        # Validate using verify_topk_output
        validation = verify_topk_output(x, result)
        assert bool(validation.all()), (
            f"bitonic_top_k with indices failed for shape {shape}, dtype {dtype}: "
            f"{int(validation.sum())}/{num_tokens} rows passed validation"
        )
    else:
        # Validate values only
        pallas_values = result[0]

        # Generate XLA reference
        xla_result = jax.vmap(lambda y: jax.lax.top_k(y, k))(x)
        xla_values = xla_result[0]

        # Sort results to compare (handles ties)
        pallas_sorted = jnp.sort(pallas_values, axis=-1)[:, ::-1]
        xla_sorted = jnp.sort(xla_values, axis=-1)[:, ::-1]

        assert jnp.allclose(pallas_sorted, xla_sorted), (
            f"bitonic_top_k failed for shape {shape}, dtype {dtype}\n"
            f"First row Pallas: {pallas_values[0, :10]}\n"
            f"First row XLA:    {xla_values[0, :10]}"
        )


def test_bitonic_top_k_invalid_shapes():
    """Test that invalid shapes raise ValueError."""
    with pytest.raises(ValueError, match="bitonic_top_k requires num_tokens <= NUM_LANES"):
        x = jnp.zeros((NUM_LANES + 1, NUM_LANES))
        bitonic_top_k(x)

    with pytest.raises(ValueError, match="vocab_size must be multiple of NUM_LANES"):
        x = jnp.zeros((NUM_LANES, NUM_LANES + 1))
        bitonic_top_k(x)

    with pytest.raises(ValueError, match="bitonic_top_k only supports k=NUM_LANES"):
        x = jnp.zeros((NUM_LANES, NUM_LANES))
        bitonic_top_k(x, k=1)
