"""
Verify CSE (Common Subexpression Elimination) on bitonic_sort for (16, 1024) input.

This script:
1. Extracts the jaxpr from bitonic_sort_arrays
2. Counts primitives in the original jaxpr
3. Applies CSE transformation
4. Counts primitives in the CSE'd jaxpr
5. Verifies correctness by comparing outputs
"""

import jax
import jax.numpy as jnp
from jax.extend.core import Jaxpr, Var, JaxprEqn, ClosedJaxpr, Literal
from collections import Counter
import hashlib

from tallax._src.bitonic_topk import bitonic_sort_arrays


def _hash_eqn(eqn):
    """Create a hashable key for a JaxprEqn."""
    invars_repr = ""
    for var in eqn.invars:
        if isinstance(var, Literal):
            invars_repr += str(var.val)
        else:
            invars_repr += str(var)
    params_repr = "".join(map(str, eqn.params.values()))
    return hashlib.md5((str(eqn.primitive) + invars_repr + params_repr).encode()).hexdigest()


def cse_jaxpr(jaxpr: Jaxpr) -> Jaxpr:
    """Performs common subexpression elimination on a Jaxpr."""
    new_eqns = []
    cse_cache = {}
    substitutions = {}

    for eqn in jaxpr.eqns:
        # Recursively apply CSE to nested jaxprs
        new_params = {}
        for k, v in eqn.params.items():
            if isinstance(v, Jaxpr):
                new_params[k] = cse_jaxpr(v)
            elif isinstance(v, ClosedJaxpr):
                new_params[k] = ClosedJaxpr(cse_jaxpr(v.jaxpr), v.consts)
            else:
                new_params[k] = v

        # Substitute inputs to the current equation
        new_invars = []
        for var in eqn.invars:
            if isinstance(var, Var):
                new_invars.append(substitutions.get(var, var))
            else:
                new_invars.append(var)
        new_eqn = eqn.replace(invars=new_invars, params=new_params)

        eqn_hash = _hash_eqn(new_eqn)

        if eqn_hash in cse_cache:
            # If we've seen this exact computation before, substitute the output
            # of this equation with the output of the cached equation.
            for out_var, cached_out_var in zip(new_eqn.outvars, cse_cache[eqn_hash].outvars):
                substitutions[out_var] = cached_out_var
        else:
            # This is a new computation, add it to our list of equations
            # and cache it.
            cse_cache[eqn_hash] = new_eqn
            new_eqns.append(new_eqn)

    # Substitute the outputs of the jaxpr
    new_outvars = [substitutions.get(var, var) for var in jaxpr.outvars]

    return Jaxpr(
        constvars=jaxpr.constvars,
        invars=jaxpr.invars,
        outvars=new_outvars,
        eqns=new_eqns,
    )


def count_primitives(jaxpr: Jaxpr) -> Counter:
    """Count primitives in a jaxpr recursively."""
    counts = Counter()

    for eqn in jaxpr.eqns:
        counts[str(eqn.primitive)] += 1

        # Recursively count in nested jaxprs
        for k, v in eqn.params.items():
            if isinstance(v, Jaxpr):
                counts.update(count_primitives(v))
            elif isinstance(v, ClosedJaxpr):
                counts.update(count_primitives(v.jaxpr))

    return counts


def main():
    # Test input shape
    shape = (16, 1024)
    print(f"Testing bitonic_sort with shape {shape}")
    print("=" * 80)

    # Generate test data
    key = jax.random.PRNGKey(42)
    arr = jax.random.normal(key, shape, dtype=jnp.float32)

    # Get the jaxpr of bitonic_sort_arrays
    print("\n1. Extracting jaxpr from bitonic_sort_arrays...")
    jaxpr_func = jax.make_jaxpr(
        lambda x: bitonic_sort_arrays([x], num_keys=1, axis=1, descending=False)
    )(arr)

    original_jaxpr = jaxpr_func.jaxpr

    # Count primitives in original jaxpr
    print("\n2. Counting primitives in original jaxpr...")
    original_counts = count_primitives(original_jaxpr)
    total_original = sum(original_counts.values())

    print(f"\nOriginal primitive counts (total: {total_original}):")
    for prim, count in sorted(original_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {prim}: {count}")

    # Apply CSE
    print("\n3. Applying CSE transformation...")
    cse_jaxpr_result = cse_jaxpr(original_jaxpr)

    # Count primitives in CSE'd jaxpr
    print("\n4. Counting primitives in CSE'd jaxpr...")
    cse_counts = count_primitives(cse_jaxpr_result)
    total_cse = sum(cse_counts.values())

    print(f"\nCSE'd primitive counts (total: {total_cse}):")
    for prim, count in sorted(cse_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {prim}: {count}")

    # Calculate reduction
    print("\n" + "=" * 80)
    print(f"RESULTS:")
    print(f"  Original total primitives: {total_original}")
    print(f"  CSE'd total primitives:    {total_cse}")
    print(f"  Reduction:                 {total_original - total_cse} ({100 * (total_original - total_cse) / total_original:.2f}%)")
    print("=" * 80)

    # Show per-primitive reductions
    print("\nPer-primitive reductions:")
    all_prims = set(original_counts.keys()) | set(cse_counts.keys())
    reductions = []
    for prim in all_prims:
        orig = original_counts.get(prim, 0)
        cse = cse_counts.get(prim, 0)
        if orig != cse:
            reductions.append((prim, orig, cse, orig - cse))

    for prim, orig, cse, reduction in sorted(reductions, key=lambda x: -abs(x[3]))[:15]:
        pct = 100 * reduction / orig if orig > 0 else 0
        print(f"  {prim:30s}: {orig:6d} -> {cse:6d} ({reduction:+6d}, {pct:+6.2f}%)")

    # Verify correctness by evaluating both jaxprs
    print("\n5. Verifying correctness...")
    try:
        # Use eval_jaxpr to evaluate the jaxprs
        from jax.core import eval_jaxpr

        # Evaluate original jaxpr
        original_result = eval_jaxpr(original_jaxpr, jaxpr_func.consts, arr)

        # Evaluate CSE'd jaxpr
        cse_result = eval_jaxpr(cse_jaxpr_result, jaxpr_func.consts, arr)

        # Compare results
        if isinstance(original_result, (list, tuple)):
            original_output = original_result[0]
            cse_output = cse_result[0]
        else:
            original_output = original_result
            cse_output = cse_result

        # Check if outputs match
        if jnp.allclose(original_output, cse_output, rtol=1e-5, atol=1e-5):
            print("✓ Verification PASSED: CSE'd jaxpr produces identical output!")
        else:
            print("✗ Verification FAILED: Outputs differ!")
            print(f"  Max difference: {jnp.max(jnp.abs(original_output - cse_output))}")

        # Also verify against reference sort
        expected = jnp.sort(arr, axis=1)
        if jnp.allclose(cse_output, expected, rtol=1e-5, atol=1e-5):
            print("✓ Correctness PASSED: Output matches reference sort!")
        else:
            print("✗ Correctness FAILED: Output doesn't match reference!")

    except Exception as e:
        print(f"✗ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
