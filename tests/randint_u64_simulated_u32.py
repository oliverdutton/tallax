import pytest
import jax
import jax.numpy as jnp
from jax.experimental import enable_x64

from tallax.vllm.high_precision_uint import modulo_u128_u64, U48, sample_random_u128_in_u32s

SHAPES = [(1, 1), (4, 4), (8, 8)]
SUM_SHAPES = [(1, 128), (2, 64), (4, 32)]
N_VALS = [128]


def _u64_from_u32s(high, low):
    """Reconstruct u64 array from (high, low) u32 pair."""
    with enable_x64():
        scale = jnp.array(2**32, dtype=jnp.uint64)
        return high.astype(jnp.uint64) * scale + low.astype(jnp.uint64)


def _max_val_to_u64_in_u32s(max_val):
    """Convert an int max_val to [high, low] u32 pair for use as modulo divisor."""
    with enable_x64():
        max_val_u64 = jnp.array(max_val, dtype=jnp.uint64)
        scale = jnp.array(2**32, dtype=jnp.uint64)
        return [
            (max_val_u64 // scale).astype(jnp.uint32),
            (max_val_u64 % scale).astype(jnp.uint32),
        ]


@pytest.mark.parametrize("shape", SHAPES)
def test_modulo_u128_u64(shape):
    """modulo_u128_u64 matches jax.random.randint in i64."""
    key = jax.random.key(0)
    max_val = 2**2
    random_u128 = sample_random_u128_in_u32s(key, shape)
    result = modulo_u128_u64(random_u128, _max_val_to_u64_in_u32s(max_val))

    with enable_x64():
        result_u64 = _u64_from_u32s(*result)
        reference = jax.random.randint(key, shape, 0, max_val, jnp.int64)
        assert (reference == result_u64).all()


@pytest.mark.parametrize("shape", SUM_SHAPES)
def test_u48_sum_overflow(shape):
    """U48 sum overflowing u32 matches x64 sum."""
    vals = jax.random.randint(jax.random.key(1), shape, 2**29, 2**30, dtype=jnp.int32)
    u48_sum = U48(vals, max_val=2**30).sum(axis=1, keepdims=True)
    sum_high, sum_low = u48_sum.to_u64_in_u32s()

    with enable_x64():
        actual = _u64_from_u32s(sum_high, sum_low)
        expected = vals.astype(jnp.int64).sum(axis=1, keepdims=True).astype(jnp.uint64)
        assert (actual == expected).all()


@pytest.mark.parametrize("n_vals", N_VALS)
def test_combined_u48_sum_modulo(n_vals):
    """U48 sum as modulo divisor matches jax.random.randint in i64.

    Uses values in [0, 2^22) so total < 2^32, where jax.random.randint
    uses the same u128 % maxval algorithm as our implementation.
    """
    vals_key, mod_key = jax.random.split(jax.random.key(42))
    vals = jax.random.randint(vals_key, (1, n_vals), 0, 2**22, dtype=jnp.int32)

    u48_sum = U48(vals, max_val=2**22).sum(axis=1, keepdims=True)
    sum_high, sum_low = u48_sum.to_u64_in_u32s()

    random_u128 = sample_random_u128_in_u32s(mod_key, (1, 1))
    result = modulo_u128_u64(random_u128, [sum_high, sum_low])

    with enable_x64():
        result_u64 = _u64_from_u32s(*result)
        total = vals.astype(jnp.int64).sum(axis=1, keepdims=True)
        reference = jax.random.randint(mod_key, (1, 1), jnp.zeros_like(total), total, jnp.int64)
        assert (result_u64 == reference).all()


@pytest.mark.parametrize("shape", SHAPES)
def test_u48_roundtrip_u64_in_u32s(shape):
    """U48 round-trips through to_u64_in_u32s / from_u64_in_u32s."""
    vals = jax.random.randint(jax.random.key(7), shape, 0, 2**31 - 1, dtype=jnp.int32)
    u48_orig = U48(vals, max_val=2**31 - 1)
    u64_u32s = u48_orig.to_u64_in_u32s()
    u48_rt = U48.from_u64_in_u32s(u64_u32s)

    # to_u64_in_u32s matches x64 conversion
    with enable_x64():
        assert (_u64_from_u32s(*u64_u32s) == vals.astype(jnp.uint64)).all()

    # from_u64_in_u32s recovers identical parts
    for a, b in zip(u48_orig.parts, u48_rt.parts):
        assert (a == b).all()


@pytest.mark.parametrize("n_vals", N_VALS)
def test_end_to_end_sum_and_modulo(n_vals):
    """End-to-end: U48 sum -> to_u32s -> modulo matches x64 sum + randint.

    Uses values in [0, 2^22) so total < 2^32, where jax.random.randint
    uses the same u128 % maxval algorithm as our implementation.
    """
    vals_key, mod_key = jax.random.split(jax.random.key(99))
    vals = jax.random.randint(vals_key, (1, n_vals), 0, 2**22, dtype=jnp.int32)

    # U32-simulated path
    u48_sum = U48(vals, max_val=2**22).sum(axis=1, keepdims=True)
    sum_high, sum_low = u48_sum.to_u64_in_u32s()
    random_u128 = sample_random_u128_in_u32s(mod_key, (1, 1))
    result = modulo_u128_u64(random_u128, [sum_high, sum_low])

    # x64 reference path
    with enable_x64():
        # Verify U48 sum matches x64 sum
        expected_sum = vals.astype(jnp.int64).sum(axis=1, keepdims=True)
        assert (_u64_from_u32s(sum_high, sum_low) == expected_sum.astype(jnp.uint64)).all()

        # Verify modulo matches jax.random.randint in i64
        result_u64 = _u64_from_u32s(*result)
        reference = jax.random.randint(mod_key, (1, 1), jnp.zeros_like(expected_sum), expected_sum, jnp.int64)
        assert (reference == result_u64).all()
