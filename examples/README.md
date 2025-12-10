# Tallax Examples

This directory contains examples demonstrating various features of custom partitioning in JAX with Shardy sharding rules.

## Examples

### 1. Scalar Array Handling

**Files:**
- `scalar_partitioning_demo.py` - Comprehensive demonstration of scalar vs sharded parameters
- `scalar_partitioning_simple.py` - Direct comparison between scalar and array parameters

**What they demonstrate:**
- How scalar (rank-0) arrays are handled in custom partitioning
- Shardy sharding rule syntax for scalars: `'b c, -> b c'` (comma with no dimension)
- PartitionSpec for scalars: `P()` means replicated on all devices
- No cross-device communication needed for scalar parameters

**Key insights:**
- Scalars are automatically replicated to all devices
- Scalars don't affect output sharding (only shaped arrays do)
- Best for global hyperparameters and constants

**Run:**
```bash
python examples/scalar_partitioning_demo.py
python examples/scalar_partitioning_simple.py
```

### 2. Axis Index in Custom Partitioning

**File:** `axis_index_custom_partitioning.py`

**What it demonstrates:**
- Using `jax.lax.axis_index(axis_name)` within custom partitioning lowering
- Per-device behavior that's aware of which device it's running on
- How different devices can produce different outputs based on their position
- Custom partitioning with JIT compilation

**Key insights:**
- `axis_index` returns the device's position (0, 1, 2, ...) along a sharded axis
- Multiple sharded dimensions accumulate their axis indices independently
- Useful for operations that need to know their position in the mesh (e.g., global indexing)
- Must use `jax.jit` to force custom partitioning lowering

**Example output:**
When sharding a 4×8 array on both batch and column axes across 4 devices:
- Device (0,0): adds 0 (batch_index=0, col_index=0)
- Device (0,1): adds 100 (batch_index=0, col_index=1)
- Device (1,0): adds 100 (batch_index=1, col_index=0)
- Device (1,1): adds 200 (batch_index=1, col_index=1)

**Run:**
```bash
python examples/axis_index_custom_partitioning.py
```

## Setup

All examples use simulated 4-device CPU clusters for testing:

```python
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

# Re-initialize JAX backend
import jax._src.xla_bridge as xb
xb.get_backend.cache_clear()
```

This allows testing sharding behavior without actual multi-device hardware.

## Key Concepts

### Custom Partitioning

Custom partitioning allows you to define how JAX operations are split across devices:

```python
@custom_partitioning
def my_operation(x, y):
    return compute(x, y)

my_operation.def_partition(
    infer_sharding_from_operands=infer_fn,
    partition=partition_fn,
    sharding_rule='b v, -> b k',  # Shardy-style rule
)
```

### Shardy Sharding Rules

Einstein-like notation describing input/output partitioning:
- `'b v, k -> b k'` - Two array inputs, one output
- `'b v, -> b k'` - Array input, scalar input (note comma), one output
- Dimensions map to PartitionSpec axes

### PartitionSpec

Describes how arrays are sharded:
- `P('batch', 'vocab')` - Shard both dimensions
- `P('batch', None)` - Shard first dim, replicate second
- `P()` - Replicate on all devices (used for scalars)

### Partitioning Callbacks

Two required callbacks:

1. **`infer_sharding_from_operands`**: Determines output sharding from inputs
2. **`partition`**: Defines per-device computation and cross-device communication

## Real-World Usage in Tallax

See `tallax/_src/sampling.py:187-245` for production examples:
- Top-K sampling with custom partitioning
- Handling both sharded logits and batched k values
- Cross-device all-gather operations
- Integration with Pallas kernels

## Further Reading

- JAX Custom Partitioning: https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html
- Main documentation: `../SCALAR_ARRAYS_CUSTOM_PARTITIONING.md`
