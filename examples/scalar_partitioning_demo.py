"""
Demonstration of scalar JAX arrays in custom partitioning with Shardy sharding rules.

This example shows:
1. How scalar (rank-0) arrays are handled differently from sharded arrays
2. Custom partitioning with Shardy-style sharding rules
3. Running on a simulated 4-device CPU cluster
"""

import jax
import jax.numpy as jnp
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import functools


def create_simple_custom_op():
    """
    Create a custom partitioned operation that demonstrates scalar handling.

    Operation: element-wise multiply array by a scalar, then sum across columns.
    - Input 1 (x): 2D array that will be sharded across devices
    - Input 2 (scale): scalar that will be replicated on all devices
    - Output: 1D array (row sums) sharded across devices
    """

    def _multiply_and_sum(x: jax.Array, scale: jax.Array):
        """Core computation: x * scale, then sum across columns."""
        return jnp.sum(x * scale, axis=1)

    @custom_partitioning
    @functools.wraps(_multiply_and_sum)
    def sharded_multiply_and_sum(x, scale):
        return _multiply_and_sum(x, scale)

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
        """
        Infer output sharding from input sharding.

        Key point: The scalar (arg_shapes[1]) has no sharding spec,
        so we only use the first input's sharding for the output.
        """
        x_spec = arg_shapes[0].sharding.spec
        print(f"  [Infer] Input x sharding spec: {x_spec}")
        print(f"  [Infer] Input scale shape: {arg_shapes[1].shape} (scalar)")

        # Output is 1D, sharded along the batch dimension (first dim of x)
        out_spec = P(x_spec[0],)
        print(f"  [Infer] Output sharding spec: {out_spec}")

        return NamedSharding(mesh, out_spec),

    def partition(mesh, arg_shapes, out_shapes):
        """
        Define how the computation is partitioned across devices.

        Key point: The scalar 'scale' is replicated (not sharded),
        so each device gets the full scalar value.
        """
        arg_shardings, out_shardings = jax.tree.map(
            lambda s: s.sharding, (arg_shapes, out_shapes)
        )

        print(f"  [Partition] arg_shardings[0].spec (x): {arg_shardings[0].spec}")
        print(f"  [Partition] arg_shardings[1].spec (scale): {arg_shardings[1].spec}")
        print(f"  [Partition] out_shardings[0].spec: {out_shardings[0].spec}")

        # Check if the second dimension of x is sharded
        x_col_axis = arg_shardings[0].spec[1] if len(arg_shardings[0].spec) > 1 else None

        def shmap_fn(x, scale):
            """Per-device computation."""
            print(f"    [Device compute] x.shape={x.shape}, scale.shape={scale.shape}")

            result = _multiply_and_sum(x, scale)

            # If columns are sharded, we need to all-reduce the sums
            if x_col_axis is not None:
                print(f"    [Device compute] Columns sharded on '{x_col_axis}', doing all-reduce")
                result = jax.lax.psum(result, x_col_axis)
            else:
                print(f"    [Device compute] Columns not sharded, no reduction needed")

            return result

        return mesh, shmap_fn, out_shardings, arg_shardings

    # Shardy sharding rule:
    # "b c, -> b" means:
    #   - First input: batch×columns (2D array)
    #   - Second input: scalar (no dimensions - note the comma before arrow)
    #   - Output: batch dimension only (1D array)
    sharded_multiply_and_sum.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule='b c, -> b',
    )

    return sharded_multiply_and_sum


def main():
    """Run demonstration on simulated 4-device CPU cluster."""

    print("="*70)
    print("SCALAR ARRAY HANDLING IN CUSTOM PARTITIONING WITH SHARDY")
    print("="*70)
    print()

    # Create simulated 4-device mesh
    # JAX can simulate multiple devices even with one physical CPU
    # Set up 4 simulated CPU devices
    import os
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

    # Re-initialize JAX backend with simulated devices
    import jax._src.xla_bridge as xb
    xb.get_backend.cache_clear()

    devices = jax.devices('cpu')[:4]  # Use 4 simulated CPUs
    print(f"Using devices: {devices}")
    print()

    # Create a 2D mesh: 2 devices for batch, 2 for columns
    import numpy as np
    mesh = Mesh(np.array(devices).reshape(2, 2), axis_names=('batch', 'col'))
    print(f"Mesh shape: {mesh.shape}")
    print(f"Mesh axis names: {mesh.axis_names}")
    print()

    # Create test data
    batch_size = 8
    num_cols = 16
    x = jnp.arange(batch_size * num_cols, dtype=jnp.float32).reshape(batch_size, num_cols)
    scale = jnp.array(2.0)  # Scalar

    print(f"Input x shape: {x.shape}")
    print(f"Input scale: {scale} (shape: {scale.shape}, ndim: {scale.ndim})")
    print()

    # Create the custom operation
    custom_op = create_simple_custom_op()

    # Test Case 1: Sharding only batch dimension
    print("-"*70)
    print("TEST CASE 1: Shard only batch dimension (scalar replicated)")
    print("-"*70)

    with mesh:
        # Shard x along batch dimension only, replicate along columns
        x_sharded = jax.device_put(x, NamedSharding(mesh, P('batch', None)))
        # Scalar is automatically replicated
        scale_sharded = jax.device_put(scale, NamedSharding(mesh, P()))

        print(f"x sharding: {x_sharded.sharding.spec}")
        print(f"scale sharding: {scale_sharded.sharding.spec}")
        print()

        result1 = custom_op(x_sharded, scale_sharded)
        print(f"\nResult shape: {result1.shape}")
        print(f"Result sharding: {result1.sharding.spec}")
        print(f"Result: {result1}")
        print()

    # Test Case 2: Sharding both dimensions
    print("-"*70)
    print("TEST CASE 2: Shard both batch and column dimensions")
    print("-"*70)

    with mesh:
        # Shard x along both dimensions
        x_sharded = jax.device_put(x, NamedSharding(mesh, P('batch', 'col')))
        # Scalar is still replicated (P() means replicated on all devices)
        scale_sharded = jax.device_put(scale, NamedSharding(mesh, P()))

        print(f"x sharding: {x_sharded.sharding.spec}")
        print(f"scale sharding: {scale_sharded.sharding.spec}")
        print()

        result2 = custom_op(x_sharded, scale_sharded)
        print(f"\nResult shape: {result2.shape}")
        print(f"Result sharding: {result2.sharding.spec}")
        print(f"Result: {result2}")
        print()

    # Verify correctness
    print("="*70)
    print("VERIFICATION")
    print("="*70)
    expected = jnp.sum(x * scale, axis=1)
    print(f"Expected result: {expected}")
    print(f"Test case 1 matches: {jnp.allclose(result1, expected)}")
    print(f"Test case 2 matches: {jnp.allclose(result2, expected)}")
    print()

    # Show key insights
    print("="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
1. Scalar arrays in custom partitioning:
   - Have PartitionSpec() - empty spec means replicated on all devices
   - Don't appear in the Shardy sharding rule dimensions (note the comma in 'b c, -> b')
   - Are automatically broadcast to all devices that need them

2. Shardy sharding rule syntax:
   - 'b c, -> b' means inputs are (2D batched array, scalar) -> 1D output
   - The comma with no dimension after it indicates a scalar parameter
   - Compare to 'b c, s -> b' which would mean a 1D array parameter 's'

3. Partitioning callbacks:
   - infer_sharding_from_operands: determines output sharding from inputs
   - partition: defines per-device computation and cross-device communication
   - Scalars don't affect sharding inference (only shaped arrays do)

4. Per-device behavior:
   - Each device receives the full scalar value
   - Sharded dimensions are split across devices
   - Reductions (like psum) needed when aggregating across sharded dimensions
""")


if __name__ == "__main__":
    main()
