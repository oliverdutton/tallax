"""Demonstration of CSE on bitonic_sort - shows optimization without full execution."""

import jax
import jax.numpy as jnp
from jax import make_jaxpr
from tallax._src.bitonic_sort import bitonic_sort_arrays
from tallax._src.cse import cse_until_fixpoint

def demo_cse():
    """Demonstrate CSE optimization on bitonic_sort jaxpr."""

    shape = (8, 1024)
    print(f"\n{'='*80}")
    print(f"CSE Demonstration on bitonic_sort (shape {shape})")
    print(f"{'='*80}\n")

    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, shape, dtype=jnp.float32)

    print(f"Input shape: {x.shape}")

    # Extract jaxpr from bitonic_sort_arrays
    print(f"\n{'-'*80}")
    print("Step 1: Extract Jaxpr from bitonic_sort_arrays")
    print(f"{'-'*80}\n")

    def sort_fn(arr):
        return bitonic_sort_arrays(
            [arr],
            num_keys=1,
            descending=False,
            max_num_fused_stages=None,
            unroll_stages=True,
            tile_unroll=None,
        )

    print("Tracing bitonic_sort_arrays to jaxpr...")
    closed_jaxpr = make_jaxpr(sort_fn)(x)
    original_eqns = len(closed_jaxpr.jaxpr.eqns)

    print(f"✓ Original jaxpr extracted")
    print(f"  Total equations: {original_eqns:,}")

    # Apply CSE
    print(f"\n{'-'*80}")
    print("Step 2: Apply Common Subexpression Elimination")
    print(f"{'-'*80}\n")

    print("Running CSE optimization...")
    cse_jaxpr, iterations = cse_until_fixpoint(closed_jaxpr.jaxpr, max_iterations=10)

    print(f"✓ CSE completed in {iterations} iterations")
    print(f"  Optimized equations: {len(cse_jaxpr.eqns):,}")
    eliminated = original_eqns - len(cse_jaxpr.eqns)
    reduction_pct = (eliminated / original_eqns) * 100

    # Results
    print(f"\n{'='*80}")
    print("CSE OPTIMIZATION RESULTS")
    print(f"{'='*80}\n")

    print(f"Input shape:           {shape}")
    print(f"Original equations:    {original_eqns:,}")
    print(f"CSE'd equations:       {len(cse_jaxpr.eqns):,}")
    print(f"Eliminated:            {eliminated:,} equations")
    print(f"Reduction:             {reduction_pct:.1f}%")

    print(f"\n✓ SUCCESS: CSE eliminated {reduction_pct:.1f}% of redundant computations!")
    print(f"\n{'='*80}\n")

    return True


if __name__ == "__main__":
    success = demo_cse()
    exit(0 if success else 1)
