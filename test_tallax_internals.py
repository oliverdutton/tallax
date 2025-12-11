"""Test tallax internal functions to isolate the segfault."""
import jax
import jax.numpy as jnp
from tallax._src.sort import _sort_pallas_vmem, sort
from tallax._src.utils import float_to_sortable_int, pad

def test_direct_sort_pallas_vmem_single():
    """Test _sort_pallas_vmem with single array (no indices)."""
    print("Test 1: _sort_pallas_vmem with single float32 array (no return_argsort)")

    key = jax.random.key(0)
    x = jax.random.normal(key, (16, 128), dtype=jnp.float32)

    # Convert to sortable int as the sort function expects
    x_int = float_to_sortable_int(x)

    # Pad as required
    x_int_padded = pad(x_int, block_shape=(128, 'power_of_2_lanes'), prepend=(False, False))

    print(f"  Input shape: {x.shape}, padded shape: {x_int_padded.shape}")

    try:
        result = _sort_pallas_vmem(
            [x_int_padded],
            num_keys=1,
            return_argsort=False,
            descending=False,
            is_stable=False,
            interpret=True
        )
        print(f"✓ SUCCESS: result shape = {result[0].shape}\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_direct_sort_pallas_vmem_with_indices():
    """Test _sort_pallas_vmem with two arrays (with indices)."""
    print("Test 2: _sort_pallas_vmem with float32 + indices (return_argsort)")
    print("WARNING: This might segfault...")

    key = jax.random.key(0)
    x = jax.random.normal(key, (16, 128), dtype=jnp.float32)

    # Convert to sortable int
    x_int = float_to_sortable_int(x)

    # Add indices
    indices = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)

    # Pad both
    x_int_padded = pad(x_int, block_shape=(128, 'power_of_2_lanes'), prepend=(False, False))
    indices_padded = pad(indices, block_shape=(128, 'power_of_2_lanes'), prepend=(False, False))

    print(f"  Input shapes: x={x.shape}, idx={indices.shape}")
    print(f"  Padded shapes: x={x_int_padded.shape}, idx={indices_padded.shape}")

    try:
        result = _sort_pallas_vmem(
            [x_int_padded, indices_padded],
            num_keys=1,  # Only sort by first array
            return_argsort=True,
            descending=False,
            is_stable=False,
            interpret=True
        )
        print(f"✓ SUCCESS: result shapes = {[r.shape for r in result]}\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_public_sort_api():
    """Test the public sort API."""
    print("Test 3: Public sort() API with float32 + return_argsort")
    print("WARNING: This is what the test uses and might segfault...")

    key = jax.random.key(0)
    x = jax.random.normal(key, (16, 128), dtype=jnp.float32)

    print(f"  Input shape: {x.shape}")

    try:
        result = sort(
            [x],
            num_keys=1,
            return_argsort=True,
            descending=False,
            is_stable=False,
            interpret=True
        )
        print(f"✓ SUCCESS: result shapes = {[r.shape for r in result]}\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Tallax Internal Functions")
    print("=" * 70)
    print()

    results = []
    results.append(("_sort_pallas_vmem single array", test_direct_sort_pallas_vmem_single()))

    # These might segfault
    print("\n" + "=" * 70)
    print("POTENTIALLY DANGEROUS TESTS")
    print("=" * 70)
    print()

    # Uncomment to test:
    # results.append(("_sort_pallas_vmem with indices", test_direct_sort_pallas_vmem_with_indices()))
    results.append(("Public sort() API", test_public_sort_api()))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
