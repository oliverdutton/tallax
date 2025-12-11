"""Test Pallas with scratch buffers and conversions like tallax does."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def float_to_sortable_int_simple(x):
    """Simplified version of float_to_sortable_int without standardize."""
    i = x.view(jnp.int32)
    return jnp.where(i < 0, i ^ 0x7FFFFFFF, i)

def sortable_int_to_float_simple(i):
    """Simplified version of sortable_int_to_float."""
    return jnp.where(i < 0, i ^ 0x7FFFFFFF, i).view(jnp.float32)

def test_with_scratch_buffers():
    """Test with scratch buffers like tallax uses."""
    print("Test: Pallas with scratch buffers (VMEM)")

    def kernel(x_ref, idx_ref, out_x_ref, out_idx_ref, scratch_x_ref, scratch_idx_ref):
        # Copy to scratch
        scratch_x_ref[...] = x_ref[...]
        scratch_idx_ref[...] = idx_ref[...]

        # Sort
        x = scratch_x_ref[...]
        idx = scratch_idx_ref[...]
        sorted_x, sorted_idx = jax.lax.sort([x, idx], num_keys=1)

        # Write output
        out_x_ref[...] = sorted_x
        out_idx_ref[...] = sorted_idx

    shape = (16, 128)
    key = jax.random.key(0)
    x = jax.random.normal(key, shape, dtype=jnp.float32)
    idx = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    try:
        result = pl.pallas_call(
            kernel,
            out_shape=[
                jax.ShapeDtypeStruct(shape, jnp.float32),
                jax.ShapeDtypeStruct(shape, jnp.int32)
            ],
            scratch_shapes=(
                pltpu.VMEM(shape, jnp.float32),
                pltpu.VMEM(shape, jnp.int32),
            ),
            interpret=True
        )(x, idx)
        print(f"✓ SUCCESS: result shapes = {[r.shape for r in result]}\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False

def test_with_float_to_int_conversion():
    """Test with float-to-int conversion like tallax uses."""
    print("Test: Pallas with float-to-int conversion")

    def kernel(x_ref, idx_ref, out_x_ref, out_idx_ref, scratch_x_ref, scratch_idx_ref):
        # Copy and convert float to sortable int
        x = x_ref[...]
        x_as_int = float_to_sortable_int_simple(x)
        scratch_x_ref[...] = x_as_int

        # Copy indices
        scratch_idx_ref[...] = idx_ref[...]

        # Sort as ints
        x_int = scratch_x_ref[...]
        idx = scratch_idx_ref[...]
        sorted_x_int, sorted_idx = jax.lax.sort([x_int, idx], num_keys=1)

        # Convert back to float
        sorted_x = sortable_int_to_float_simple(sorted_x_int)

        # Write output
        out_x_ref[...] = sorted_x
        out_idx_ref[...] = sorted_idx

    shape = (16, 128)
    key = jax.random.key(0)
    x = jax.random.normal(key, shape, dtype=jnp.float32)
    idx = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    try:
        result = pl.pallas_call(
            kernel,
            out_shape=[
                jax.ShapeDtypeStruct(shape, jnp.float32),
                jax.ShapeDtypeStruct(shape, jnp.int32)
            ],
            scratch_shapes=(
                pltpu.VMEM(shape, jnp.int32),
                pltpu.VMEM(shape, jnp.int32),
            ),
            interpret=True
        )(x, idx)
        print(f"✓ SUCCESS: result shapes = {[r.shape for r in result]}\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False

def test_with_blockspec():
    """Test with BlockSpec like tallax uses."""
    print("Test: Pallas with BlockSpec")

    def kernel(x_ref, idx_ref, out_x_ref, out_idx_ref):
        x = x_ref[...]
        idx = idx_ref[...]
        sorted_x, sorted_idx = jax.lax.sort([x, idx], num_keys=1)
        out_x_ref[...] = sorted_x
        out_idx_ref[...] = sorted_idx

    shape = (16, 128)
    key = jax.random.key(0)
    x = jax.random.normal(key, shape, dtype=jnp.float32)
    idx = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    block_shape = (16, 128)

    try:
        result = pl.pallas_call(
            kernel,
            in_specs=[
                pl.BlockSpec(block_shape, lambda i, j: (i, j)),
                pl.BlockSpec(block_shape, lambda i, j: (i, j))
            ],
            out_shape=[
                jax.ShapeDtypeStruct(shape, jnp.float32),
                jax.ShapeDtypeStruct(shape, jnp.int32)
            ],
            out_specs=[
                pl.BlockSpec(block_shape, lambda i, j: (i, j)),
                pl.BlockSpec(block_shape, lambda i, j: (i, j))
            ],
            grid=(1, 1),
            interpret=True
        )(x, idx)
        print(f"✓ SUCCESS: result shapes = {[r.shape for r in result]}\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False

def test_combined_complex():
    """Test combining all the features: BlockSpec + scratch + conversion."""
    print("Test: Combined (BlockSpec + scratch + float-to-int conversion)")
    print("WARNING: This is closest to tallax implementation...")

    def kernel(x_ref, idx_ref, out_x_ref, out_idx_ref, scratch_x_ref, scratch_idx_ref):
        # Convert float to sortable int
        x = x_ref[...]
        x_as_int = float_to_sortable_int_simple(x)
        scratch_x_ref[...] = x_as_int
        scratch_idx_ref[...] = idx_ref[...]

        # Sort
        x_int = scratch_x_ref[...]
        idx = scratch_idx_ref[...]
        sorted_x_int, sorted_idx = jax.lax.sort([x_int, idx], num_keys=1)

        # Convert back and write
        out_x_ref[...] = sortable_int_to_float_simple(sorted_x_int)
        out_idx_ref[...] = sorted_idx

    shape = (16, 128)
    key = jax.random.key(0)
    x = jax.random.normal(key, shape, dtype=jnp.float32)
    idx = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    block_shape = (16, 128)

    try:
        result = pl.pallas_call(
            kernel,
            in_specs=[
                pl.BlockSpec(block_shape, lambda i, j: (i, j)),
                pl.BlockSpec(block_shape, lambda i, j: (i, j))
            ],
            out_shape=[
                jax.ShapeDtypeStruct(shape, jnp.float32),
                jax.ShapeDtypeStruct(shape, jnp.int32)
            ],
            out_specs=[
                pl.BlockSpec(block_shape, lambda i, j: (i, j)),
                pl.BlockSpec(block_shape, lambda i, j: (i, j))
            ],
            scratch_shapes=(
                pltpu.VMEM(block_shape, jnp.int32),
                pltpu.VMEM(block_shape, jnp.int32),
            ),
            grid=(1, 1),
            interpret=True
        )(x, idx)
        print(f"✓ SUCCESS: result shapes = {[r.shape for r in result]}\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("Pallas Scratch Buffer & Conversion Tests")
    print("=" * 70)
    print()

    results = []
    results.append(("Scratch buffers (VMEM)", test_with_scratch_buffers()))
    results.append(("Float-to-int conversion", test_with_float_to_int_conversion()))
    results.append(("BlockSpec", test_with_blockspec()))
    results.append(("Combined (all features)", test_combined_complex()))

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
