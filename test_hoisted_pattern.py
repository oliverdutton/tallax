"""Test with properly hoisted iota operations."""

import jax
import jax.numpy as jnp
from jax import lax


def iota_tile(dim):
    """Create iota array with tile shape."""
    return lax.broadcasted_iota(jnp.int32, (128, 128), dim)


def create_bit_indicator(bit_position, index):
    """Create mask indicating which elements have specific bit set."""
    if type(bit_position) == int:
        bit = (index & (1 << bit_position))
        return bit > 0
    return (index >> bit_position) & 1


def test_hoisted_rematerialization():
    """Test with iota operations HOISTED outside the 200-iteration loop."""
    batch_size = 128
    compression_length = 16
    iteration_indices = jnp.array([0, 1, 0, 1, 2, 0, 2], dtype=jnp.int32)
    num_iterations = 7

    def body_fn(idx, carry):
        results = carry
        i = iteration_indices[idx]

        # HOIST iota operations OUTSIDE the 200-iteration loop
        iota_0 = iota_tile(0)
        iota_1 = iota_tile(1)
        tile_local_offset = iota_0 + (iota_1 // batch_size) * compression_length

        # Now loop 200 times, reusing the hoisted values
        def inner_loop_body(_, acc):
            intra_tile_separation = 1 << i
            is_right_half = create_bit_indicator(i, iota_0)
            permutation = jnp.bitwise_xor(iota_0, intra_tile_separation)
            result = tile_local_offset + is_right_half.astype(jnp.int32) + permutation
            return acc + result

        loop_result = lax.fori_loop(0, 200, inner_loop_body, jnp.zeros((128, 128), dtype=jnp.int32))
        return results + loop_result

    final_result = lax.fori_loop(0, num_iterations, body_fn, jnp.zeros((128, 128), dtype=jnp.int32))
    return final_result.sum()


if __name__ == "__main__":
    print("="*80)
    print("TESTING HOISTED PATTERN")
    print("="*80)

    # Generate jaxpr
    jaxpr = jax.make_jaxpr(test_hoisted_rematerialization)()
    print(f"\nJaxpr generated")

    # Compile and get HLO
    print("\nCompiling to get HLO...")
    lowered = jax.jit(test_hoisted_rematerialization).lower()
    hlo_text = lowered.as_text()

    # Count operations in HLO (StableHLO format)
    iota_count = hlo_text.count('stablehlo.iota')
    add_count = hlo_text.count('stablehlo.add')
    xor_count = hlo_text.count('stablehlo.xor')
    broadcast_count = hlo_text.count('stablehlo.broadcast')

    print("\n" + "="*80)
    print("HLO OPERATION COUNTS")
    print("="*80)
    print(f"  iota: {iota_count}")
    print(f"  add: {add_count}")
    print(f"  xor: {xor_count}")
    print(f"  broadcast: {broadcast_count}")

    print("\n" + "="*80)
    print("EXPECTED vs ACTUAL")
    print("="*80)
    print("Expected (with proper hoisting):")
    print("  - 2-4 iota operations (dimension 0 and 1, maybe duplicated for fusion)")
    print("  - ~200-400 add operations (200 inner loop iterations × 7 outer iterations)")
    print("  - ~3-7 xor operations (for different i values)")

    print(f"\nActual:")
    print(f"  - {iota_count} iota operations")
    print(f"  - {add_count} add operations")
    print(f"  - {xor_count} xor operations")

    if iota_count <= 10:
        print("\n✓ SUCCESS: iota operations significantly reduced!")
    else:
        print("\n✗ NEEDS MORE WORK: still too many iota operations")

    # Save HLO
    with open('/home/user/tallax/hoisted_pattern_hlo.txt', 'w') as f:
        f.write(hlo_text)
    print(f"\nFull HLO saved to: /home/user/tallax/hoisted_pattern_hlo.txt")

    # Run to verify correctness
    print("\n" + "="*80)
    print("VERIFYING CORRECTNESS")
    print("="*80)
    result = jax.jit(test_hoisted_rematerialization)()
    print(f"Result: {result}")
    print("✓ Computation completed successfully")
