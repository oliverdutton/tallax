"""
Improved CSE verification for bitonic_sort.

Improvements:
1. Counts primitives recursively through jit primitives
2. Runs CSE iteratively until fixpoint
3. Tests multiple input shapes including (16, 32768)
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


def count_primitives(jaxpr: Jaxpr, recurse_through_jit: bool = True) -> Counter:
    """Count primitives in a jaxpr recursively.

    Args:
        jaxpr: The jaxpr to count primitives in
        recurse_through_jit: If True, recurse into jit primitives instead of counting them

    Returns:
        Counter of primitive names to counts
    """
    counts = Counter()

    for eqn in jaxpr.eqns:
        prim_name = str(eqn.primitive)

        # Check if this is a jit primitive with a nested jaxpr
        if prim_name == 'jit' and recurse_through_jit:
            # Don't count the jit itself, recurse through it
            found_nested = False
            for param_key in ['jaxpr', 'call_jaxpr']:
                if param_key in eqn.params:
                    v = eqn.params[param_key]
                    if isinstance(v, Jaxpr):
                        counts.update(count_primitives(v, recurse_through_jit=True))
                        found_nested = True
                        break
                    elif isinstance(v, ClosedJaxpr):
                        counts.update(count_primitives(v.jaxpr, recurse_through_jit=True))
                        found_nested = True
                        break

            if not found_nested:
                # Count the jit if we can't recurse
                counts[prim_name] += 1
        else:
            counts[prim_name] += 1

            # Recursively count in other nested jaxprs
            for k, v in eqn.params.items():
                if isinstance(v, Jaxpr):
                    counts.update(count_primitives(v, recurse_through_jit=recurse_through_jit))
                elif isinstance(v, ClosedJaxpr):
                    counts.update(count_primitives(v.jaxpr, recurse_through_jit=recurse_through_jit))

    return counts


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


def cse_until_fixpoint(jaxpr: Jaxpr, max_iterations: int = 10) -> tuple[Jaxpr, int]:
    """Apply CSE repeatedly until no more changes occur.

    Args:
        jaxpr: The jaxpr to apply CSE to
        max_iterations: Maximum number of iterations

    Returns:
        Tuple of (final jaxpr, number of iterations)
    """
    iterations = 0
    current_jaxpr = jaxpr

    for i in range(max_iterations):
        iterations += 1
        new_jaxpr = cse_jaxpr(current_jaxpr)

        # Check if anything changed by comparing equation counts
        current_count = count_primitives(current_jaxpr, recurse_through_jit=True)
        new_count = count_primitives(new_jaxpr, recurse_through_jit=True)

        if current_count == new_count:
            # No change, we've reached a fixed point
            break

        current_jaxpr = new_jaxpr

    return current_jaxpr, iterations


def test_shape(shape):
    """Test CSE on bitonic_sort_arrays with given shape."""
    print(f"\nTesting bitonic_sort with shape {shape}")
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

    # Count primitives in original jaxpr (without recursing through jit)
    print("\n2. Counting primitives (with jit counted)...")
    original_counts_with_jit = count_primitives(original_jaxpr, recurse_through_jit=False)
    total_original_with_jit = sum(original_counts_with_jit.values())
    print(f"Total primitives (counting jit): {total_original_with_jit}")
    print(f"Number of jit primitives: {original_counts_with_jit.get('jit', 0)}")

    # Count primitives in original jaxpr (recursing through jit)
    print("\n3. Counting primitives (recursing through jit)...")
    original_counts = count_primitives(original_jaxpr, recurse_through_jit=True)
    total_original = sum(original_counts.values())

    print(f"\nOriginal primitive counts (total: {total_original}):")
    for prim, count in sorted(original_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {prim}: {count}")

    # Apply CSE once
    print("\n4. Applying CSE transformation once...")
    cse_jaxpr_once = cse_jaxpr(original_jaxpr)
    cse_counts_once = count_primitives(cse_jaxpr_once, recurse_through_jit=True)
    total_cse_once = sum(cse_counts_once.values())
    print(f"After 1 iteration: {total_cse_once} primitives ({total_original - total_cse_once} reduction)")

    # Apply CSE to fixpoint
    print("\n5. Applying CSE until fixpoint...")
    fixpoint_jaxpr, iterations = cse_until_fixpoint(original_jaxpr)
    fixpoint_counts = count_primitives(fixpoint_jaxpr, recurse_through_jit=True)
    total_fixpoint = sum(fixpoint_counts.values())

    print(f"\nFixpoint reached in {iterations} iterations")
    print(f"Fixpoint primitive counts (total: {total_fixpoint}):")
    for prim, count in sorted(fixpoint_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {prim}: {count}")

    # Calculate reduction
    print("\n" + "=" * 80)
    print(f"RESULTS:")
    print(f"  Original total primitives:     {total_original}")
    print(f"  After 1 CSE iteration:         {total_cse_once} ({100 * (total_original - total_cse_once) / total_original:.2f}%)")
    print(f"  After fixpoint ({iterations} iterations): {total_fixpoint} ({100 * (total_original - total_fixpoint) / total_original:.2f}%)")
    print(f"  Additional reduction at fixpoint: {total_cse_once - total_fixpoint}")
    print("=" * 80)

    # Show per-primitive reductions (fixpoint vs original)
    print("\nPer-primitive reductions (fixpoint vs original):")
    all_prims = set(original_counts.keys()) | set(fixpoint_counts.keys())
    reductions = []
    for prim in all_prims:
        orig = original_counts.get(prim, 0)
        fixed = fixpoint_counts.get(prim, 0)
        if orig != fixed:
            reductions.append((prim, orig, fixed, orig - fixed))

    for prim, orig, fixed, reduction in sorted(reductions, key=lambda x: -abs(x[3]))[:20]:
        pct = 100 * reduction / orig if orig > 0 else 0
        print(f"  {prim:30s}: {orig:6d} -> {fixed:6d} ({reduction:+6d}, {pct:+6.2f}%)")

    # Verify correctness
    print("\n6. Verifying correctness...")
    try:
        from jax.core import eval_jaxpr

        # Evaluate original jaxpr
        original_result = eval_jaxpr(original_jaxpr, jaxpr_func.consts, arr)

        # Evaluate fixpoint jaxpr
        fixpoint_result = eval_jaxpr(fixpoint_jaxpr, jaxpr_func.consts, arr)

        # Compare results
        if isinstance(original_result, (list, tuple)):
            original_output = original_result[0]
            fixpoint_output = fixpoint_result[0]
        else:
            original_output = original_result
            fixpoint_output = fixpoint_result

        # Check if outputs match
        if jnp.allclose(original_output, fixpoint_output, rtol=1e-5, atol=1e-5):
            print("✓ Verification PASSED: Fixpoint jaxpr produces identical output!")
        else:
            print("✗ Verification FAILED: Outputs differ!")
            print(f"  Max difference: {jnp.max(jnp.abs(original_output - fixpoint_output))}")

        # Also verify against reference sort
        expected = jnp.sort(arr, axis=1)
        if jnp.allclose(fixpoint_output, expected, rtol=1e-5, atol=1e-5):
            print("✓ Correctness PASSED: Output matches reference sort!")
        else:
            print("✗ Correctness FAILED: Output doesn't match reference!")

    except Exception as e:
        print(f"✗ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()

    return {
        'shape': shape,
        'original': total_original,
        'cse_once': total_cse_once,
        'fixpoint': total_fixpoint,
        'iterations': iterations,
        'reduction_pct': 100 * (total_original - total_fixpoint) / total_original
    }


def main():
    # Test multiple shapes
    shapes = [(16, 1024), (16, 32768)]

    results = []
    for shape in shapes:
        result = test_shape(shape)
        results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for r in results:
        print(f"\nShape {r['shape']}:")
        print(f"  Original:  {r['original']:6d} primitives")
        print(f"  1 iter:    {r['cse_once']:6d} primitives ({100 * (r['original'] - r['cse_once']) / r['original']:5.2f}%)")
        print(f"  Fixpoint:  {r['fixpoint']:6d} primitives ({r['reduction_pct']:5.2f}%) in {r['iterations']} iterations")


if __name__ == "__main__":
    main()
