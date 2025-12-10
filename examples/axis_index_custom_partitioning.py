"""
Demonstration of using axis_index within custom partitioning.

This example shows how to add the device's axis index to outputs within
a custom partitioned function, demonstrating per-device behavior.
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


def create_axis_aware_operation():
    """
    Create a custom partitioned operation that adds axis indices to outputs.

    This demonstrates:
    1. How to use jax.lax.axis_index within custom partitioning
    2. Per-device computation that's aware of which device it's on
    3. Different outputs from different devices based on their position
    """

    def _compute(x: jax.Array) -> jax.Array:
        """Base computation: just return x unchanged."""
        return x * 2.0

    @custom_partitioning
    @functools.wraps(_compute)
    def axis_aware_compute(x):
        """Wrapped version that will add axis indices when sharded."""
        return _compute(x)

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
        """Output has same sharding as input."""
        print("\n[Infer Sharding]")
        x_spec = arg_shapes[0].sharding.spec
        print(f"  Input sharding spec: {x_spec}")
        print(f"  Output sharding spec: {x_spec}")
        return NamedSharding(mesh, x_spec),

    def partition(mesh, arg_shapes, out_shapes):
        """
        Define per-device computation that adds axis indices.

        Key: Use jax.lax.axis_index to get the device's position
        along each sharded axis, then add it to the output.
        """
        arg_shardings, out_shardings = jax.tree.map(
            lambda s: s.sharding, (arg_shapes, out_shapes)
        )

        print("\n[Partition]")
        x_spec = arg_shardings[0].spec
        print(f"  Input sharding spec: {x_spec}")

        # Extract which axes are sharded
        sharded_axes = {}
        for dim_idx, axis_name in enumerate(x_spec):
            if axis_name is not None:
                sharded_axes[dim_idx] = axis_name
                print(f"  Dimension {dim_idx} sharded on axis '{axis_name}'")

        if not sharded_axes:
            print("  No sharded axes - will run without axis index additions")

        def shmap_fn(x):
            """
            Per-device computation.

            This function runs on each device independently.
            We add the axis index to demonstrate per-device behavior.
            """
            result = _compute(x)

            # Add axis indices for each sharded dimension
            for dim_idx, axis_name in sharded_axes.items():
                # Get this device's index along the axis
                idx = jax.lax.axis_index(axis_name)

                # Add axis index to all elements
                # Multiply by 100 to make it clearly visible in output
                result = result + (idx * 100)

            return result

        return mesh, shmap_fn, out_shardings, arg_shardings

    axis_aware_compute.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule='b c -> b c',  # 2D input -> 2D output
    )

    return axis_aware_compute


def main():
    print("="*70)
    print("AXIS INDEX IN CUSTOM PARTITIONING")
    print("="*70)

    devices = jax.devices('cpu')[:4]
    print(f"\nUsing {len(devices)} simulated CPU devices: {devices}\n")

    # Create a 2x2 mesh
    mesh = Mesh(np.array(devices).reshape(2, 2), axis_names=('batch', 'col'))
    print(f"Mesh shape: {mesh.shape}")
    print(f"Mesh axes: {mesh.axis_names}\n")

    # Create test data
    batch_size = 4
    num_cols = 8
    x = jnp.arange(batch_size * num_cols, dtype=jnp.float32).reshape(batch_size, num_cols)

    print(f"Input shape: {x.shape}")
    print(f"Input data:")
    print(x)
    print()

    # Create the custom operation
    axis_aware_op = create_axis_aware_operation()

    # Test Case 1: No sharding (replicated)
    print("="*70)
    print("TEST CASE 1: No sharding (replicated)")
    print("="*70)
    print("Expected: Output = x * 2.0 (no axis index added)")

    with mesh:
        x_replicated = jax.device_put(x, NamedSharding(mesh, P(None, None)))
        print(f"\nInput sharding: {x_replicated.sharding.spec}")

        # JIT compile to force custom partitioning lowering
        jitted_op = jax.jit(axis_aware_op)
        result1 = jitted_op(x_replicated)

        print(f"\nResult shape: {result1.shape}")
        print(f"Result sharding: {result1.sharding.spec}")
        print(f"Result:")
        print(result1)
        print(f"\nExpected (x * 2.0):")
        print(x * 2.0)
        print(f"Match: {jnp.allclose(result1, x * 2.0)}")

    # Test Case 2: Shard only batch dimension
    print("\n" + "="*70)
    print("TEST CASE 2: Shard only batch dimension")
    print("="*70)
    print("Expected: Output = x * 2.0 + (batch_axis_index * 100)")

    with mesh:
        x_batch_sharded = jax.device_put(x, NamedSharding(mesh, P('batch', None)))
        print(f"\nInput sharding: {x_batch_sharded.sharding.spec}")

        # JIT compile to force custom partitioning lowering
        jitted_op = jax.jit(axis_aware_op)
        result2 = jitted_op(x_batch_sharded)

        print(f"\nResult shape: {result2.shape}")
        print(f"Result sharding: {result2.sharding.spec}")
        print(f"Result:")
        print(result2)

        # Build expected result
        expected2 = x * 2.0
        # Rows 0-1 are on batch_axis_index=0, rows 2-3 are on batch_axis_index=1
        expected2 = expected2.at[0:2].add(0 * 100)  # First batch shard
        expected2 = expected2.at[2:4].add(1 * 100)  # Second batch shard

        print(f"\nExpected (x * 2.0 + batch_index * 100):")
        print(expected2)
        print(f"Match: {jnp.allclose(result2, expected2)}")

    # Test Case 3: Shard only column dimension
    print("\n" + "="*70)
    print("TEST CASE 3: Shard only column dimension")
    print("="*70)
    print("Expected: Output = x * 2.0 + (col_axis_index * 100)")

    with mesh:
        x_col_sharded = jax.device_put(x, NamedSharding(mesh, P(None, 'col')))
        print(f"\nInput sharding: {x_col_sharded.sharding.spec}")

        # JIT compile to force custom partitioning lowering
        jitted_op = jax.jit(axis_aware_op)
        result3 = jitted_op(x_col_sharded)

        print(f"\nResult shape: {result3.shape}")
        print(f"Result sharding: {result3.sharding.spec}")
        print(f"Result:")
        print(result3)

        # Build expected result
        expected3 = x * 2.0
        # Cols 0-3 are on col_axis_index=0, cols 4-7 are on col_axis_index=1
        expected3 = expected3.at[:, 0:4].add(0 * 100)  # First col shard
        expected3 = expected3.at[:, 4:8].add(1 * 100)  # Second col shard

        print(f"\nExpected (x * 2.0 + col_index * 100):")
        print(expected3)
        print(f"Match: {jnp.allclose(result3, expected3)}")

    # Test Case 4: Shard both dimensions
    print("\n" + "="*70)
    print("TEST CASE 4: Shard both batch and column dimensions")
    print("="*70)
    print("Expected: Output = x * 2.0 + (batch_index * 100) + (col_index * 100)")

    with mesh:
        x_fully_sharded = jax.device_put(x, NamedSharding(mesh, P('batch', 'col')))
        print(f"\nInput sharding: {x_fully_sharded.sharding.spec}")

        # JIT compile to force custom partitioning lowering
        jitted_op = jax.jit(axis_aware_op)
        result4 = jitted_op(x_fully_sharded)

        print(f"\nResult shape: {result4.shape}")
        print(f"Result sharding: {result4.sharding.spec}")
        print(f"Result:")
        print(result4)

        # Build expected result
        # Each quadrant gets different additions based on its position in the mesh
        expected4 = x * 2.0
        expected4 = expected4.at[0:2, 0:4].add(0 * 100 + 0 * 100)  # Device (0,0)
        expected4 = expected4.at[0:2, 4:8].add(0 * 100 + 1 * 100)  # Device (0,1)
        expected4 = expected4.at[2:4, 0:4].add(1 * 100 + 0 * 100)  # Device (1,0)
        expected4 = expected4.at[2:4, 4:8].add(1 * 100 + 1 * 100)  # Device (1,1)

        print(f"\nExpected (x * 2.0 + batch_index * 100 + col_index * 100):")
        print(expected4)
        print(f"Match: {jnp.allclose(result4, expected4)}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Test Case 1 (No sharding):      {"✓ PASS" if jnp.allclose(result1, x * 2.0) else "✗ FAIL"}
Test Case 2 (Batch sharded):    {"✓ PASS" if jnp.allclose(result2, expected2) else "✗ FAIL"}
Test Case 3 (Column sharded):   {"✓ PASS" if jnp.allclose(result3, expected3) else "✗ FAIL"}
Test Case 4 (Fully sharded):    {"✓ PASS" if jnp.allclose(result4, expected4) else "✗ FAIL"}

Key Insights:
1. jax.lax.axis_index(axis_name) returns this device's index along that axis
2. Different devices can produce different outputs based on their position
3. Axis indices start at 0 and increment for each device along that axis
4. Multiple sharded dimensions accumulate their axis indices independently
5. When not sharded (replicated), no axis index is added
""")


if __name__ == "__main__":
    main()
