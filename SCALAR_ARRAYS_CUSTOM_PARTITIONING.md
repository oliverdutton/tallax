# Scalar Arrays in Custom Partitioning with Shardy Sharding Rules

This document explains how scalar (rank-0) JAX arrays are handled in custom partitioning when using Shardy-style sharding rules.

## Summary

**Key Point**: Scalar arrays in custom partitioning are automatically replicated across all devices and are indicated in Shardy sharding rules by a comma with no following dimension.

## Examples

Two working examples demonstrate this behavior:

1. **`examples/scalar_partitioning_demo.py`** - Complete demonstration with element-wise operations
2. **`examples/scalar_partitioning_simple.py`** - Direct comparison between scalar and array parameters

Both examples run on a simulated 4-CPU cluster and verify correct behavior.

## Shardy Sharding Rule Syntax

The sharding rule is an Einstein-like notation that describes how inputs and outputs are partitioned:

```python
# Array parameter (can be sharded)
sharding_rule='b v, k -> b k'
# Input 1: batch × vocab (2D)
# Input 2: k (1D array)
# Output: batch × k (2D)

# Scalar parameter (always replicated)
sharding_rule='b v, -> b k'
# Input 1: batch × vocab (2D)
# Input 2: (scalar - note comma with nothing after)
# Output: batch × k (2D)
```

**The comma with no dimensions indicates a scalar parameter.**

## Real-World Example from Tallax

In `tallax/_src/sampling.py:187-245`, the `top_k` function demonstrates scalar handling:

```python
@custom_partitioning
def sharded_top_k(logits, k):
    return _top_k(logits, k)

sharded_top_k.def_partition(
    infer_sharding_from_operands=infer_sharding_from_operands,
    partition=partition,
    sharding_rule='b v, b -> b k, b k',  # k is per-batch, not scalar
)
```

Note: In this case, `k` is actually a **batched array** (one k value per batch element), not a scalar. If it were scalar, the rule would be `'b v, -> b k, b k'`.

The `replace_val` parameter (line 187) is a true scalar, but it's handled as a closure variable rather than an explicit input, so it doesn't appear in the sharding rule.

## PartitionSpec for Scalars

Scalars use an empty `PartitionSpec()`:

```python
# Scalar: replicated on all devices
scalar_sharding = NamedSharding(mesh, P())

# 1D array: sharded along axis 'x'
array_sharding = NamedSharding(mesh, P('x'))

# 2D array: sharded along both axes
matrix_sharding = NamedSharding(mesh, P('batch', 'vocab'))
```

## Custom Partitioning Callbacks

Two callbacks define how operations are partitioned:

### 1. `infer_sharding_from_operands`

Determines output sharding from input sharding:

```python
def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
    # Get sharding spec from first (non-scalar) input
    x_spec = arg_shapes[0].sharding.spec

    # Scalar input: arg_shapes[1].shape == ()
    # Scalar has no spec to consider

    # Output takes sharding from x
    return NamedSharding(mesh, x_spec),
```

**Key insight**: Scalars don't influence output sharding - only shaped arrays do.

### 2. `partition`

Defines per-device computation and communication:

```python
def partition(mesh, arg_shapes, out_shapes):
    arg_shardings, out_shardings = jax.tree.map(
        lambda s: s.sharding, (arg_shapes, out_shapes)
    )

    # Check if we need cross-device communication
    x_axis = arg_shardings[0].spec[1]  # Could be None or an axis name

    def shmap_fn(x, scalar):
        # scalar is replicated - all devices get the same value
        result = compute(x, scalar)

        # If sharded, reduce across devices
        if x_axis is not None:
            result = jax.lax.psum(result, x_axis)

        return result

    return mesh, shmap_fn, out_shardings, arg_shardings
```

## Device Distribution

When running on a 4-device cluster:

### Sharded Array (e.g., shape `(8, 16)` sharded as `P('batch', 'col')`):
```
Device 0: rows 0-3, cols 0-7
Device 1: rows 0-3, cols 8-15
Device 2: rows 4-7, cols 0-7
Device 3: rows 4-7, cols 8-15
```

### Scalar (e.g., value `2.0` with `P()`):
```
Device 0: 2.0
Device 1: 2.0
Device 2: 2.0
Device 3: 2.0
```

All devices receive the complete scalar value - no communication needed!

## Practical Implications

### When to Use Scalars

1. **Global hyperparameters**: temperature, top_p (when same for all tokens)
2. **Constants**: replacement values, scaling factors
3. **Single configuration values**: random seeds, flags

### When to Use Arrays

1. **Per-element parameters**: per-token temperatures
2. **Batched inputs**: different k values per batch element
3. **Values that should be sharded**: large tensors

### Performance Considerations

| Aspect | Scalar | Array |
|--------|--------|-------|
| Memory per device | Minimal (one value) | Proportional to shard size |
| Communication | None (replicated once) | May require gather/scatter |
| Sharding flexibility | Always replicated | Can be sharded or replicated |

## Common Patterns in Tallax

### Pattern 1: Static scalar arguments
```python
def operation(x, static_value):
    # Use functools.partial or closure to capture static_value
    # Don't include it as a traced argument
```

### Pattern 2: Replicated scalar arguments
```python
sharding_rule='b v, -> b k'  # Scalar after comma
# Scalar is traced but replicated on all devices
```

### Pattern 3: Batched scalar-like arguments
```python
sharding_rule='b v, b -> b k'  # One value per batch
# Shape (batch,) - not a true scalar, but one value per example
```

## Testing on Simulated Cluster

Both example files use JAX's device simulation:

```python
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

# Re-initialize to pick up the flag
import jax._src.xla_bridge as xb
xb.get_backend.cache_clear()

devices = jax.devices('cpu')[:4]  # Now returns 4 simulated devices
```

This allows testing sharding behavior without access to actual multi-device hardware.

## Running the Examples

```bash
# Install dependencies
pip install -e .

# Run the comprehensive demo
python examples/scalar_partitioning_demo.py

# Run the scalar vs array comparison
python examples/scalar_partitioning_simple.py
```

Both examples should print verification that results match expected values.

## Key Takeaways

1. **Sharding rule syntax**: Comma with nothing after = scalar parameter
2. **PartitionSpec**: Empty `P()` = replicated scalar
3. **Inference**: Scalars don't affect output sharding
4. **Partitioning**: Scalars are automatically replicated to all devices
5. **Communication**: No cross-device communication needed for scalars
6. **Use case**: Best for global constants and hyperparameters

## References

- JAX custom partitioning: https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html
- Shardy (JAX's sharding system): Part of JAX internal infrastructure
- Tallax implementation: `tallax/_src/sampling.py:187-245`
