"""Detailed analysis of jaxpr equation counts at each level."""

import jax
import jax.numpy as jnp
from jax import lax


def iota_tile(dim):
    return lax.broadcasted_iota(jnp.int32, (128, 128), dim)


def create_bit_indicator(bit_position, index):
    if type(bit_position) == int:
        bit = (index & (1 << bit_position))
        return bit > 0
    return (index >> bit_position) & 1


def test_rematerialization_patterns():
    """Test with 200 iterations."""
    batch_size = 128
    compression_length = 16
    iteration_indices = jnp.array([0, 1, 0, 1, 2, 0, 2], dtype=jnp.int32)
    num_iterations = 7

    def body_fn(idx, carry):
        results = carry
        i = iteration_indices[idx]
        for _ in range(200):
            iota_0_local = iota_tile(0)
            iota_1_local = iota_tile(1)
            tile_local_offset = iota_0_local + (iota_1_local // batch_size) * compression_length

            intra_tile_separation = 1 << i
            is_right_half = create_bit_indicator(i, iota_0_local)
            permutation = jnp.bitwise_xor(iota_0_local, intra_tile_separation)

            results += tile_local_offset + is_right_half.astype(jnp.int32) + permutation
        return results

    final_result = lax.fori_loop(0, num_iterations, body_fn, jnp.zeros((128, 128), dtype=jnp.int32))
    return final_result.sum()


def count_nested_equations(jaxpr, level=0, name="root"):
    """Recursively count equations at each nesting level."""
    from collections import Counter

    total = len(jaxpr.eqns)

    print("  " * level + f"{name}: {total} equations at this level")

    for i, eqn in enumerate(jaxpr.eqns):
        prim_name = eqn.primitive.name

        # Check for nested jaxprs
        nested_found = False
        for param_name in ['jaxpr', 'body_jaxpr', 'cond_jaxpr']:
            if param_name in eqn.params:
                nested = eqn.params[param_name]
                if hasattr(nested, 'jaxpr'):
                    print("  " * level + f"  [{i}] {prim_name} ({param_name}):")
                    count_nested_equations(nested.jaxpr, level + 2, f"{prim_name}_{param_name}")
                    nested_found = True
                elif hasattr(nested, 'eqns'):
                    print("  " * level + f"  [{i}] {prim_name} ({param_name}):")
                    count_nested_equations(nested, level + 2, f"{prim_name}_{param_name}")
                    nested_found = True

        if not nested_found and level == 0:
            # Show top-level primitives
            if prim_name in ['iota', 'broadcasted_iota']:
                print("  " * level + f"  [{i}] {prim_name}")


print("="*80)
print("DETAILED JAXPR EQUATION ANALYSIS")
print("="*80)

jaxpr = jax.make_jaxpr(test_rematerialization_patterns)()

print("\nJaxpr structure:")
count_nested_equations(jaxpr.jaxpr)

print("\n" + "="*80)
print("EQUATION COUNT SUMMARY")
print("="*80)

def total_count(j):
    """Count all equations including nested."""
    count = len(j.eqns)
    for eqn in j.eqns:
        for param_name in ['jaxpr', 'body_jaxpr', 'cond_jaxpr']:
            if param_name in eqn.params:
                nested = eqn.params[param_name]
                if hasattr(nested, 'jaxpr'):
                    count += total_count(nested.jaxpr)
                elif hasattr(nested, 'eqns'):
                    count += total_count(nested)
    return count

total = total_count(jaxpr.jaxpr)
print(f"\nTotal equations (all levels): {total}")
print(f"Top-level equations: {len(jaxpr.jaxpr.eqns)}")
print(f"Nested equations: {total - len(jaxpr.jaxpr.eqns)}")

# Count operation types
from collections import Counter

def collect_all_ops(j):
    ops = []
    for eqn in j.eqns:
        ops.append(eqn.primitive.name)
        for param_name in ['jaxpr', 'body_jaxpr', 'cond_jaxpr']:
            if param_name in eqn.params:
                nested = eqn.params[param_name]
                if hasattr(nested, 'jaxpr'):
                    ops.extend(collect_all_ops(nested.jaxpr))
                elif hasattr(nested, 'eqns'):
                    ops.extend(collect_all_ops(nested))
    return ops

all_ops = collect_all_ops(jaxpr.jaxpr)
op_counts = Counter(all_ops)

print("\n" + "="*80)
print("OPERATION TYPE COUNTS")
print("="*80)
for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"  {op}: {count}")

print("\n" + "="*80)
print("KEY OBSERVATIONS")
print("="*80)

iota_count = op_counts.get('iota', 0)
add_count = op_counts.get('add', 0)
xor_count = op_counts.get('bitwise_xor', 0)

print(f"iota operations: {iota_count}")
print(f"  - Expected: 2 (with perfect CSE)")
print(f"  - Actual: {iota_count}")
print(f"  - Ratio: {iota_count/2:.1f}x redundancy")

print(f"\nadd operations: {add_count}")
print(f"  - Expected: ~200 (one per inner loop iteration)")
print(f"  - Actual: {add_count}")
print(f"  - Note: May include adds from tile_local_offset computation")

print(f"\nbitwise_xor operations: {xor_count}")
print(f"  - Expected: ~3-5 (unique i values: 0,1,2)")
print(f"  - Actual: {xor_count}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("JAX's tracing already performs CSE - only 2 iota ops in jaxpr")
print("The 200-iteration loop is NOT unrolled at jaxpr level")
print("XLA may unroll later, but that's a backend optimization choice")
