"""
Simple demonstration showing how scalar arrays behave in custom partitioning.

This example compares:
1. A regular sharded array parameter
2. A scalar (rank-0) parameter

Key observation: Scalars are automatically replicated across all devices,
while regular arrays can be sharded.
"""

import jax
import jax.numpy as jnp
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import functools
import os
import numpy as np

# Set up 4 simulated CPU devices
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
import jax._src.xla_bridge as xb
xb.get_backend.cache_clear()


def compare_scalar_vs_array_sharding():
    """
    Demonstrates the difference between scalar and array parameters
    in custom partitioning.
    """

    # Version 1: Both parameters are arrays (can be sharded)
    @custom_partitioning
    def add_with_array_offset(x, offset):
        """Add an array offset to each row."""
        return x + offset

    def infer_array_sharding(mesh, arg_shapes, result_shape):
        x_spec = arg_shapes[0].sharding.spec
        offset_spec = arg_shapes[1].sharding.spec
        print(f"\n[Array version - Infer]")
        print(f"  x spec: {x_spec}")
        print(f"  offset spec: {offset_spec} (can be sharded)")
        # Output has same sharding as x
        return NamedSharding(mesh, x_spec),

    def partition_array(mesh, arg_shapes, out_shapes):
        arg_shardings, out_shardings = jax.tree.map(
            lambda s: s.sharding, (arg_shapes, out_shapes)
        )
        print(f"\n[Array version - Partition]")
        print(f"  x sharding: {arg_shardings[0].spec}")
        print(f"  offset sharding: {arg_shardings[1].spec}")

        def impl(x, offset):
            return x + offset

        return mesh, impl, out_shardings, arg_shardings

    add_with_array_offset.def_partition(
        infer_sharding_from_operands=infer_array_sharding,
        partition=partition_array,
        sharding_rule='b c, c -> b c',  # Both inputs have dimensions
    )

    # Version 2: Second parameter is scalar (always replicated)
    @custom_partitioning
    def add_with_scalar_offset(x, offset):
        """Add a scalar offset to all elements."""
        return x + offset

    def infer_scalar_sharding(mesh, arg_shapes, result_shape):
        x_spec = arg_shapes[0].sharding.spec
        offset_shape = arg_shapes[1].shape
        print(f"\n[Scalar version - Infer]")
        print(f"  x spec: {x_spec}")
        print(f"  offset shape: {offset_shape} (scalar - no sharding)")
        # Output has same sharding as x
        return NamedSharding(mesh, x_spec),

    def partition_scalar(mesh, arg_shapes, out_shapes):
        arg_shardings, out_shardings = jax.tree.map(
            lambda s: s.sharding, (arg_shapes, out_shapes)
        )
        print(f"\n[Scalar version - Partition]")
        print(f"  x sharding: {arg_shardings[0].spec}")
        print(f"  offset sharding: {arg_shardings[1].spec} (always replicated)")

        def impl(x, offset):
            return x + offset

        return mesh, impl, out_shardings, arg_shardings

    add_with_scalar_offset.def_partition(
        infer_sharding_from_operands=infer_scalar_sharding,
        partition=partition_scalar,
        sharding_rule='b c, -> b c',  # Note: comma with no dimensions = scalar
    )

    return add_with_array_offset, add_with_scalar_offset


def main():
    print("="*70)
    print("SCALAR vs ARRAY PARAMETERS IN CUSTOM PARTITIONING")
    print("="*70)

    devices = jax.devices('cpu')[:4]
    print(f"\nUsing {len(devices)} simulated CPU devices")
    mesh = Mesh(np.array(devices).reshape(2, 2), axis_names=('batch', 'col'))
    print(f"Mesh: {mesh.shape}")

    # Create test data
    x = jnp.arange(4 * 8, dtype=jnp.float32).reshape(4, 8)
    array_offset = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    scalar_offset = jnp.array(10.0)

    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  array_offset: {array_offset.shape} (1D array)")
    print(f"  scalar_offset: {scalar_offset.shape} (scalar, ndim={scalar_offset.ndim})")

    add_with_array_offset, add_with_scalar_offset = compare_scalar_vs_array_sharding()

    # Test 1: Array offset
    print("\n" + "="*70)
    print("TEST 1: Adding array offset (can be sharded)")
    print("="*70)

    with mesh:
        x_sharded = jax.device_put(x, NamedSharding(mesh, P('batch', 'col')))
        offset_sharded = jax.device_put(array_offset, NamedSharding(mesh, P('col')))

        # Force compilation and partitioning by calling the function
        result1 = add_with_array_offset(x_sharded, offset_sharded)

        print(f"\nResult shape: {result1.shape}")
        print(f"Result sharding: {result1.sharding.spec}")
        print(f"First row: {result1[0]}")

    # Test 2: Scalar offset
    print("\n" + "="*70)
    print("TEST 2: Adding scalar offset (always replicated)")
    print("="*70)

    with mesh:
        x_sharded = jax.device_put(x, NamedSharding(mesh, P('batch', 'col')))
        # Scalar is automatically replicated - P() means replicated
        scalar_sharded = jax.device_put(scalar_offset, NamedSharding(mesh, P()))

        result2 = add_with_scalar_offset(x_sharded, scalar_sharded)

        print(f"\nResult shape: {result2.shape}")
        print(f"Result sharding: {result2.sharding.spec}")
        print(f"First row: {result2[0]}")

    # Verification
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    expected1 = x + array_offset
    expected2 = x + scalar_offset
    print(f"Array offset result matches: {jnp.allclose(result1, expected1)}")
    print(f"Scalar offset result matches: {jnp.allclose(result2, expected2)}")

    # Key insights
    print("\n" + "="*70)
    print("KEY DIFFERENCES")
    print("="*70)
    print("""
1. Sharding Rule Syntax:
   - Array parameter: 'b c, c -> b c'  (offset has dimension 'c')
   - Scalar parameter: 'b c, -> b c'   (comma with nothing = scalar)

2. PartitionSpec:
   - Array: P('col') - can be sharded along the 'col' axis
   - Scalar: P() - empty spec = replicated on all devices

3. Device Distribution:
   - Array: Different devices get different slices (e.g., columns 0-3, 4-7)
   - Scalar: All devices get the complete value (10.0)

4. Communication Cost:
   - Array: May require device-to-device transfer if resharded
   - Scalar: Minimal cost - just replicated to all devices

5. Use Cases:
   - Array parameters: When values vary per-element (e.g., per-token temperatures)
   - Scalar parameters: Global constants (e.g., single temperature for all tokens)
""")


if __name__ == "__main__":
    main()
