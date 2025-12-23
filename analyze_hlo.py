"""Extract and analyze HLO from the test pattern."""

import jax
import jax.numpy as jnp
from jax import lax
import re


def iota_tile(dim):
    """Create iota array with tile shape."""
    return lax.broadcasted_iota(jnp.int32, (128, 128), dim)


def create_bit_indicator(bit_position, index):
    """Create mask indicating which elements have specific bit set."""
    if type(bit_position) == int:
        bit = (index & (1 << bit_position))
        return bit > 0
    return (index >> bit_position) & 1


def test_rematerialization_patterns():
    """Test pattern with 200 iterations to verify CSE."""
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


print("="*80)
print("GENERATING HLO")
print("="*80)

# Compile and get HLO
compiled = jax.jit(test_rematerialization_patterns).lower().compile()

# Get HLO text
hlo_text = compiled.as_text()

print("\n" + "="*80)
print("HLO ANALYSIS")
print("="*80)

# Count operations in HLO
iota_count = len(re.findall(r'\biota\b', hlo_text))
broadcast_count = len(re.findall(r'\bbroadcast\b', hlo_text))
xor_count = len(re.findall(r'\bxor\b', hlo_text))
add_count = len(re.findall(r'\badd\b', hlo_text))
multiply_count = len(re.findall(r'\bmultiply\b', hlo_text))

print(f"\nOperation counts in HLO:")
print(f"  iota: {iota_count}")
print(f"  broadcast: {broadcast_count}")
print(f"  xor: {xor_count}")
print(f"  add: {add_count}")
print(f"  multiply: {multiply_count}")

print(f"\n" + "="*80)
print("EXPECTED vs ACTUAL")
print(f"="*80)
print("Expected (with perfect CSE):")
print("  - 2 iota operations (dimension 0 and 1)")
print("  - ~3-5 xor operations (for different i values)")
print("  - ~200 add operations (200 iterations per loop)")
print(f"\nActual:")
print(f"  - {iota_count} iota operations")
print(f"  - {xor_count} xor operations")
print(f"  - {add_count} add operations")

status = "✓ EXCELLENT" if iota_count <= 2 else "✗ NEEDS IMPROVEMENT"
print(f"\nStatus: {status}")

# Show relevant HLO sections
print(f"\n" + "="*80)
print("HLO IOTA OPERATIONS")
print(f"="*80)

lines = hlo_text.split('\n')
for i, line in enumerate(lines):
    if 'iota' in line.lower():
        # Show context
        start = max(0, i-1)
        end = min(len(lines), i+3)
        print(f"\nLines {start}-{end}:")
        for j in range(start, end):
            marker = ">>> " if j == i else "    "
            print(f"{marker}{lines[j]}")

# Save full HLO
with open('/home/user/tallax/test_pattern_hlo.txt', 'w') as f:
    f.write(hlo_text)

print(f"\n" + "="*80)
print(f"Full HLO saved to: /home/user/tallax/test_pattern_hlo.txt")
print(f"="*80)

# Also show a snippet of the computation
print(f"\n" + "="*80)
print("HLO COMPUTATION SNIPPET (first 100 lines)")
print(f"="*80)
print('\n'.join(lines[:100]))
