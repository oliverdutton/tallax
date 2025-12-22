"""Test CSE on the three key rematerialization patterns.

This creates a simple loop to verify that constants and repeated computations
are properly identified and eliminated by CSE.
"""

import jax
import jax.numpy as jnp
from jax import lax
from collections import Counter
import re


def iota_tile(dim):
    """Create iota array with tile shape - matching tallax implementation."""
    return lax.broadcasted_iota(jnp.int32, (128, 128), dim)


def create_bit_indicator(bit_position, index):
    """Create mask indicating which elements have specific bit set."""
    if type(bit_position) == int:
        bit = (index & (1 << bit_position))
        return bit > 0
    return (index >> bit_position) & 1


def test_rematerialization_patterns():
    """Test CSE removes duplicate iota_tile calls and derived computations.

    Pattern 1: tile_local_offset = iota_tile(0) + (iota_tile(1) // batch_size) * compression_length
    Pattern 2: is_right_half = create_bit_indicator(log2(intra_tile_separation), iota_tile(axis))
    Pattern 3: permutation = jnp.bitwise_xor(iota_tile(axis), intra_tile_separation)

    Key: iota_tile is called INSIDE loop - CSE must remove duplicates
    """
    batch_size = 128  # Constant
    compression_length = 16  # Constant
    # Use repeated i values: [0, 1, 0, 1, 2, 0, 2] to test CSE on duplicates
    iteration_indices = jnp.array([0, 1, 0, 1, 2, 0, 2], dtype=jnp.int32)
    num_iterations = 7

    def body_fn(idx, carry):
        results = carry
        i = iteration_indices[idx]  # Get actual iteration value

        # Pattern 1: Call iota_tile INSIDE loop (not hoisted!)
        # CSE should eliminate duplicate calls
        iota_0_local = iota_tile(0)  # Called 7 times, should be CSE'd
        iota_1_local = iota_tile(1)  # Called 7 times, should be CSE'd
        tile_local_offset = iota_0_local + (iota_1_local // batch_size) * compression_length

        # Pattern 2: is_right_half
        # Since i values repeat (0,1,0,1,2,0,2), CSE should recognize duplicates
        intra_tile_separation = 1 << i
        is_right_half = create_bit_indicator(i, iota_0_local)

        # Pattern 3: permutation
        # With repeated i values, these should also be CSE'd
        permutation = jnp.bitwise_xor(iota_0_local, intra_tile_separation)

        # Add together to create dependency
        result = tile_local_offset + is_right_half.astype(jnp.int32) + permutation
        return results + result

    final_result = lax.fori_loop(0, num_iterations, body_fn, jnp.zeros((128, 128), dtype=jnp.int32))
    return final_result.sum()


def count_operations_in_jaxpr(jaxpr_obj):
    """Count operations in jaxpr including nested ones."""
    jaxpr = jaxpr_obj.jaxpr if hasattr(jaxpr_obj, 'jaxpr') else jaxpr_obj

    ops = []
    def collect_ops(j):
        for eqn in j.eqns:
            ops.append(eqn.primitive.name)
            # Recursively check nested jaxprs
            for param_name in ['jaxpr', 'body_jaxpr', 'cond_jaxpr']:
                if param_name in eqn.params:
                    nested = eqn.params[param_name]
                    if hasattr(nested, 'jaxpr'):
                        collect_ops(nested.jaxpr)
                    elif hasattr(nested, 'eqns'):
                        collect_ops(nested)

    collect_ops(jaxpr)
    return Counter(ops)


def analyze_jaxpr_text(jaxpr_str):
    """Analyze jaxpr text representation."""
    # Count iota operations
    iota_count = len(re.findall(r'\w+:\w+\[\d+,\d+\]\s*=\s*iota\[', jaxpr_str))

    # Count unique iota signatures
    iota_ops = re.findall(r'iota\[(.*?)\n.*?\]', jaxpr_str, re.DOTALL)
    iota_signatures = Counter(iota_ops)

    return iota_count, iota_signatures


# Test without CSE
print("="*80)
print("TESTING REMATERIALIZATION PATTERNS")
print("="*80)

jaxpr_original = jax.make_jaxpr(test_rematerialization_patterns)()
ops_original = count_operations_in_jaxpr(jaxpr_original)

print(f"\nOriginal jaxpr:")
print(f"  Total equations: {len(jaxpr_original.jaxpr.eqns)}")
print(f"  iota operations: {ops_original.get('iota', 0)}")
print(f"  broadcast_in_dim: {ops_original.get('broadcast_in_dim', 0)}")
print(f"  integer_pow: {ops_original.get('integer_pow', 0)}")
print(f"  bitwise_xor: {ops_original.get('bitwise_xor', 0)}")

# Analyze text representation
jaxpr_str = str(jaxpr_original)
iota_count, iota_sigs = analyze_jaxpr_text(jaxpr_str)
print(f"\nFrom text analysis:")
print(f"  Total iota ops: {iota_count}")
print(f"  Unique iota signatures: {len(iota_sigs)}")

# Expected counts
print(f"\n" + "="*80)
print("EXPECTED vs ACTUAL (BEFORE CSE)")
print("="*80)
print("With iota_tile called 7 times INSIDE loop (no hoisting):")
print("  WITHOUT CSE: Should see 14 iota operations (7x iota(0) + 7x iota(1))")
print(f"  ACTUAL (before CSE): {iota_count}")
print(f"  Status: {' EXPECTED' if iota_count >= 14 else '⚠ LESS THAN EXPECTED'}")

# Show where iotas are
print(f"\n" + "="*80)
print("IOTA OPERATION DETAILS")
print("="*80)
print("Searching for iota operations in jaxpr...")
lines = jaxpr_str.split('\n')
for i, line in enumerate(lines):
    if 'iota[' in line:
        print(f"Line {i}: {line.strip()}")
        # Show context
        if i + 1 < len(lines):
            for j in range(1, min(4, len(lines) - i)):
                if lines[i+j].strip():
                    print(f"      {lines[i+j].strip()}")
                if ']' in lines[i+j]:
                    break

# Now apply CSE
print(f"\n" + "="*80)
print("APPLYING CSE")
print("="*80)

from iterative_cse_pass import apply_iterative_cse

optimized_jaxpr, eliminations, iterations = apply_iterative_cse(
    jaxpr_original,
    max_iterations=10,
    verbose=True
)

print(f"\n" + "="*80)
print("AFTER CSE")
print("="*80)

# Analyze optimized jaxpr
jaxpr_opt_str = str(optimized_jaxpr)
iota_count_opt, iota_sigs_opt = analyze_jaxpr_text(jaxpr_opt_str)

print(f"  iota operations: {iota_count} → {iota_count_opt}")
print(f"  WITH GOOD CSE: Should reduce to 2 (one iota(0), one iota(1))")
print(f"  Reduction: {iota_count - iota_count_opt} operations eliminated ({(1-iota_count_opt/iota_count)*100:.1f}%)")
print(f"  Status: {'✓ EXCELLENT CSE' if iota_count_opt == 2 else '✗ CSE NEEDS IMPROVEMENT'}")

if iota_count_opt > 2:
    print(f"\n  Remaining iota operations:")
    lines = jaxpr_opt_str.split('\n')
    for i, line in enumerate(lines):
        if 'iota[' in line:
            print(f"    Line {i}: {line.strip()[:100]}")
