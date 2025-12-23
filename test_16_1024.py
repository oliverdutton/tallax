"""Quick test for (16, 1024) shape."""
import sys
sys.path.insert(0, '/home/user/tallax')

import jax
import jax.numpy as jnp
from tallax._src.bitonic_sort import bitonic_sort_arrays

print("Testing bitonic_sort_arrays with shape (16, 1024)...")
shape = (16, 1024)
key = jax.random.PRNGKey(42)
arr = jax.random.randint(key, shape, 0, 1000, dtype=jnp.int32)

print(f"Input shape: {arr.shape}")

# Test basic sort
print("\n1. Testing basic sort...")
try:
    result = bitonic_sort_arrays([arr], num_keys=1, descending=False)
    sorted_arr = result[0]
    print(f"✓ Basic sort succeeded, output shape: {sorted_arr.shape}")

    # Verify sorting
    expected = jnp.sort(arr, axis=1)
    if jnp.allclose(sorted_arr, expected):
        print("✓ Sort is correct!")
    else:
        print("✗ Sort mismatch!")
        sys.exit(1)
except Exception as e:
    print(f"✗ Basic sort failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with tile_unroll=8
print("\n2. Testing with tile_unroll=8...")
try:
    result = bitonic_sort_arrays([arr], num_keys=1, descending=False, tile_unroll=8)
    sorted_arr = result[0]
    print(f"✓ tile_unroll=8 succeeded, output shape: {sorted_arr.shape}")

    if jnp.allclose(sorted_arr, expected):
        print("✓ tile_unroll=8 is correct!")
    else:
        print("✗ tile_unroll=8 mismatch!")
        sys.exit(1)
except Exception as e:
    print(f"✗ tile_unroll=8 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with max_num_fused_stages
print("\n3. Testing with max_num_fused_stages=5...")
try:
    result = bitonic_sort_arrays([arr], num_keys=1, descending=False, max_num_fused_stages=5)
    sorted_arr = result[0]
    print(f"✓ max_num_fused_stages=5 succeeded")

    if jnp.allclose(sorted_arr, expected):
        print("✓ max_num_fused_stages=5 is correct!")
    else:
        print("✗ max_num_fused_stages=5 mismatch!")
        sys.exit(1)
except Exception as e:
    print(f"✗ max_num_fused_stages=5 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with unroll_stages
print("\n4. Testing with unroll_stages=True...")
try:
    result = bitonic_sort_arrays([arr], num_keys=1, descending=False, unroll_stages=True)
    sorted_arr = result[0]
    print(f"✓ unroll_stages=True succeeded")

    if jnp.allclose(sorted_arr, expected):
        print("✓ unroll_stages=True is correct!")
    else:
        print("✗ unroll_stages=True mismatch!")
        sys.exit(1)
except Exception as e:
    print(f"✗ unroll_stages=True failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("All tests passed!")
print("="*60)
