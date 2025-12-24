import jax
import jax.numpy as jnp
from jax import make_jaxpr
from jax.extend.core import Jaxpr, Var, JaxprEqn, ClosedJaxpr, Literal, jaxpr_as_fun
from collections import Counter, defaultdict
import hashlib
import pickle
import os
import sys

# ============================================================================
# CSE IMPLEMENTATION (Inlined)
# ============================================================================

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


def cse_jaxpr(jaxpr: Jaxpr, recurse_through_jit: bool = True) -> Jaxpr:
    """Performs common subexpression elimination on a Jaxpr."""
    new_eqns = []
    cse_cache = {}
    substitutions = {}

    for eqn in jaxpr.eqns:
        # Recursively apply CSE to nested jaxprs
        new_params = {}
        for k, v in eqn.params.items():
            if isinstance(v, Jaxpr):
                new_params[k] = cse_jaxpr(v, recurse_through_jit=recurse_through_jit)
            elif isinstance(v, ClosedJaxpr):
                new_params[k] = ClosedJaxpr(
                    cse_jaxpr(v.jaxpr, recurse_through_jit=recurse_through_jit),
                    v.consts
                )
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
            for out_var, cached_out_var in zip(new_eqn.outvars, cse_cache[eqn_hash].outvars):
                substitutions[out_var] = cached_out_var
        else:
            # This is a new computation
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
    """Apply CSE repeatedly until no more changes occur."""
    iterations = 0
    current_jaxpr = jaxpr

    for i in range(max_iterations):
        iterations += 1
        new_jaxpr = cse_jaxpr(current_jaxpr)

        # Check if anything changed
        if len(new_jaxpr.eqns) == len(current_jaxpr.eqns):
            # No change, we've reached a fixed point
            break

        current_jaxpr = new_jaxpr

    return current_jaxpr, iterations


# ============================================================================
# PRIMITIVE COUNTING (Inlined)
# ============================================================================

def count_primitives(jaxpr: Jaxpr, recurse_through_jit: bool = True) -> Counter:
    """Count primitives in a jaxpr recursively."""
    counts = Counter()

    for eqn in jaxpr.eqns:
        prim_name = str(eqn.primitive)

        # Check if this is a jit primitive with a nested jaxpr
        if prim_name == 'jit' and recurse_through_jit:
            # Don't count the jit itself, recurse through it
            if 'jaxpr' in eqn.params:
                v = eqn.params['jaxpr']
                if isinstance(v, Jaxpr):
                    counts.update(count_primitives(v, recurse_through_jit=True))
                elif isinstance(v, ClosedJaxpr):
                    counts.update(count_primitives(v.jaxpr, recurse_through_jit=True))
            elif 'call_jaxpr' in eqn.params:
                v = eqn.params['call_jaxpr']
                if isinstance(v, Jaxpr):
                    counts.update(count_primitives(v, recurse_through_jit=True))
                elif isinstance(v, ClosedJaxpr):
                    counts.update(count_primitives(v.jaxpr, recurse_through_jit=True))
            else:
                counts[prim_name] += 1
        else:
            counts[prim_name] += 1

            # Recursively count in nested jaxprs
            for k, v in eqn.params.items():
                if isinstance(v, Jaxpr):
                    counts.update(count_primitives(v, recurse_through_jit=recurse_through_jit))
                elif isinstance(v, ClosedJaxpr):
                    counts.update(count_primitives(v.jaxpr, recurse_through_jit=recurse_through_jit))

    return counts


def count_primitives_by_shape(jaxpr: Jaxpr, recurse_through_jit: bool = True) -> dict:
    """Count primitives stratified by output shapes."""
    shape_stratified = defaultdict(lambda: defaultdict(int))

    def process_jaxpr(j):
        for eqn in j.eqns:
            prim_name = str(eqn.primitive)

            # Get output shape(s)
            if hasattr(eqn, 'outvars') and eqn.outvars:
                for outvar in eqn.outvars:
                    if hasattr(outvar, 'aval') and hasattr(outvar.aval, 'shape'):
                        shape = tuple(outvar.aval.shape)
                        dtype = str(outvar.aval.dtype) if hasattr(outvar.aval, 'dtype') else 'unknown'
                        shape_key = f"{shape}:{dtype}"

                        # Skip jit if recursing through it
                        if prim_name == 'jit' and recurse_through_jit:
                            pass
                        else:
                            shape_stratified[shape_key][prim_name] += 1

            # Recurse into nested jaxprs
            if prim_name != 'jit' or not recurse_through_jit:
                for k, v in eqn.params.items():
                    if isinstance(v, Jaxpr):
                        process_jaxpr(v)
                    elif isinstance(v, ClosedJaxpr):
                        process_jaxpr(v.jaxpr)
            else:
                # Recurse through jit
                if 'jaxpr' in eqn.params:
                    v = eqn.params['jaxpr']
                    if isinstance(v, Jaxpr):
                        process_jaxpr(v)
                    elif isinstance(v, ClosedJaxpr):
                        process_jaxpr(v.jaxpr)

    process_jaxpr(jaxpr)
    return dict(shape_stratified)


# ============================================================================
# JAXPR EXTRACTION (Inlined)
# ============================================================================

def extract_kernel_jaxpr(jaxpr_obj):
    """Extract the innermost kernel jaxpr (from pallas_call and run_scoped)."""
    if hasattr(jaxpr_obj, 'jaxpr'):
        jaxpr = jaxpr_obj.jaxpr
    else:
        jaxpr = jaxpr_obj

    # Navigate through jit
    for eqn in jaxpr.eqns:
        if str(eqn.primitive) == 'jit' and 'jaxpr' in eqn.params:
            jit_jaxpr = eqn.params['jaxpr']
            if isinstance(jit_jaxpr, ClosedJaxpr):
                jit_jaxpr = jit_jaxpr.jaxpr

            # Find pallas_call
            for jit_eqn in jit_jaxpr.eqns:
                if str(jit_eqn.primitive) == 'pallas_call' and 'jaxpr' in jit_eqn.params:
                    pallas_jaxpr = jit_eqn.params['jaxpr']
                    if isinstance(pallas_jaxpr, ClosedJaxpr):
                        pallas_jaxpr = pallas_jaxpr.jaxpr

                    # Check for run_scoped
                    for pallas_eqn in pallas_jaxpr.eqns:
                        if str(pallas_eqn.primitive) == 'run_scoped':
                            if 'jaxpr' in pallas_eqn.params:
                                scoped_jaxpr = pallas_eqn.params['jaxpr']
                                if isinstance(scoped_jaxpr, ClosedJaxpr):
                                    return scoped_jaxpr.jaxpr
                                return scoped_jaxpr

                    # No run_scoped, return pallas jaxpr
                    return pallas_jaxpr

    return None


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_version(version_name, jaxpr, apply_cse=False):
    """Analyze a jaxpr and return statistics."""
    if apply_cse:
        jaxpr, cse_iterations = cse_until_fixpoint(jaxpr, max_iterations=10)
    else:
        cse_iterations = 0

    counts = count_primitives(jaxpr, recurse_through_jit=True)
    shapes = count_primitives_by_shape(jaxpr, recurse_through_jit=True)

    return {
        'jaxpr': jaxpr,
        'version': version_name,
        'cse_applied': apply_cse,
        'cse_iterations': cse_iterations,
        'total_eqns': len(jaxpr.eqns),
        'total_primitives': sum(counts.values()),
        'unique_primitives': len(counts),
        'primitive_counts': counts,
        'shape_stratified': shapes,
        'unique_shapes': len(shapes),
    }

def print_analysis(result, label):
    """Print analysis results."""
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")
    print(f"Total equations:        {result['total_eqns']:,}")
    print(f"Total primitives:       {result['total_primitives']:,}")
    print(f"Unique primitive types: {result['unique_primitives']}")
    print(f"Unique shape signatures: {result['unique_shapes']}")
    if result['cse_applied']:
        print(f"CSE iterations:         {result['cse_iterations']}")

    print(f"\nTop 15 primitives:")
    for prim, count in result['primitive_counts'].most_common(15):
        pct = (count / result['total_primitives'] * 100) if result['total_primitives'] > 0 else 0
        print(f"  {prim:30s}: {count:>8,} ({pct:>5.1f}%)")


# ============================================================================
# TEST CASE WITH REDUNDANT COMPUTATIONS
# ============================================================================

def test_function_with_redundancy(x, y):
    """A simple function with obvious common subexpressions."""
    # Compute x + y multiple times (redundant)
    a = x + y
    b = x + y  # Same as a

    # Compute x * 2 multiple times (redundant)
    c = x * 2.0
    d = x * 2.0  # Same as c

    # Use the redundant computations
    result1 = a * c  # (x + y) * (x * 2)
    result2 = b * d  # (x + y) * (x * 2) - same computation!

    # More redundancy
    e = result1 + result2
    f = result1 + result2  # Same as e

    return e + f  # Should be 2 * e


def main():
    print("="*80)
    print("CSE JAXPR TEST - Testing Common Subexpression Elimination")
    print("="*80)

    # Create test inputs
    x = jnp.array(3.0)
    y = jnp.array(4.0)

    print(f"\nTest inputs: x={x}, y={y}")

    # Generate jaxpr for the test function
    print("\n[1] Generating Jaxpr from test function...")
    closed_jaxpr = make_jaxpr(test_function_with_redundancy)(x, y)
    original_jaxpr = closed_jaxpr.jaxpr

    print(f"Original Jaxpr has {len(original_jaxpr.eqns)} equations")

    # Print original jaxpr
    print("\n" + "="*80)
    print("ORIGINAL JAXPR:")
    print("="*80)
    print(original_jaxpr)

    # Analyze original
    print("\n[2] Analyzing original Jaxpr...")
    original_analysis = analyze_version("Original", original_jaxpr, apply_cse=False)
    print_analysis(original_analysis, "ORIGINAL JAXPR ANALYSIS")

    # Apply CSE
    print("\n[3] Applying CSE transformation...")
    cse_analysis = analyze_version("CSE", original_jaxpr, apply_cse=True)
    cse_jaxpr = cse_analysis['jaxpr']

    print(f"CSE'd Jaxpr has {len(cse_jaxpr.eqns)} equations")
    print(f"Reduction: {len(original_jaxpr.eqns)} -> {len(cse_jaxpr.eqns)} equations")
    print(f"Eliminated {len(original_jaxpr.eqns) - len(cse_jaxpr.eqns)} redundant computations")

    # Print CSE'd jaxpr
    print("\n" + "="*80)
    print("CSE'D JAXPR:")
    print("="*80)
    print(cse_jaxpr)

    print_analysis(cse_analysis, "CSE'D JAXPR ANALYSIS")

    # Execute both versions and compare
    print("\n" + "="*80)
    print("EXECUTION TEST - Verifying CSE preserves semantics")
    print("="*80)

    print("\n[4] Executing original function...")
    original_result = test_function_with_redundancy(x, y)
    print(f"Original result: {original_result}")

    print("\n[5] Executing original Jaxpr...")
    original_jaxpr_fn = jaxpr_as_fun(closed_jaxpr)
    original_jaxpr_result = original_jaxpr_fn(x, y)
    print(f"Original Jaxpr result: {original_jaxpr_result}")
    # Extract the single result from the list
    original_jaxpr_result = original_jaxpr_result[0]

    print("\n[6] Executing CSE'd Jaxpr...")
    # Create a ClosedJaxpr for execution
    cse_closed_jaxpr = ClosedJaxpr(cse_jaxpr, closed_jaxpr.consts)
    cse_jaxpr_fn = jaxpr_as_fun(cse_closed_jaxpr)
    cse_jaxpr_result = cse_jaxpr_fn(x, y)
    print(f"CSE'd Jaxpr result: {cse_jaxpr_result}")
    # Extract the single result from the list
    cse_jaxpr_result = cse_jaxpr_result[0]

    # Verify correctness
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    original_matches = jnp.allclose(original_result, original_jaxpr_result)
    cse_matches = jnp.allclose(original_result, cse_jaxpr_result)

    print(f"\nOriginal function == Original Jaxpr: {original_matches}")
    print(f"Original function == CSE'd Jaxpr:    {cse_matches}")

    if original_matches and cse_matches:
        print("\n✓ SUCCESS: CSE transformation preserves semantics!")
        print(f"  All three versions produce the same result: {original_result}")
    else:
        print("\n✗ FAILURE: Results don't match!")
        print(f"  Original function: {original_result}")
        print(f"  Original Jaxpr:    {original_jaxpr_result}")
        print(f"  CSE'd Jaxpr:       {cse_jaxpr_result}")
        return 1

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Input:  x={x}, y={y}")
    print(f"Output: {original_result}")
    print(f"\nOriginal equations: {len(original_jaxpr.eqns)}")
    print(f"CSE'd equations:    {len(cse_jaxpr.eqns)}")
    print(f"Reduction:          {len(original_jaxpr.eqns) - len(cse_jaxpr.eqns)} equations eliminated")
    print(f"Efficiency gain:    {(1 - len(cse_jaxpr.eqns)/len(original_jaxpr.eqns))*100:.1f}%")
    print("\n" + "="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
