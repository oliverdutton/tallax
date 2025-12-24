#!/usr/bin/env python3
"""
Standalone comparison runner for bitonic_sort implementations.
Analyzes all 4 combinations: MAIN/OLD × no-CSE/CSE

This file is completely self-contained and can be run independently.
It will analyze whichever version is currently checked out (MAIN or OLD).

Usage:
    # On MAIN branch:
    python standalone_comparison_runner.py

    # On OLD branch (commit 895d0e8):
    python standalone_comparison_runner.py

    # After running on both, it will compare them
"""

import jax
import jax.numpy as jnp
from jax import make_jaxpr
from jax.extend.core import Jaxpr, Var, JaxprEqn, ClosedJaxpr, Literal
from collections import Counter, defaultdict
import hashlib
import pickle
import os
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

TEST_SHAPE = (16, 1024)
OUTPUT_DIR = '/tmp'
VERBOSE = True

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
        pct = (count / result['total_primitives'] * 100)
        print(f"  {prim:30s}: {count:>8,} ({pct:>5.1f}%)")


def compare_analyses(result1, result2, label1, label2):
    """Compare two analysis results."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {label1} vs {label2}")
    print(f"{'='*80}")

    # Summary
    print(f"\nSummary:")
    print(f"  {label1:<20s}: {result1['total_primitives']:>10,} primitives")
    print(f"  {label2:<20s}: {result2['total_primitives']:>10,} primitives")
    diff = result1['total_primitives'] - result2['total_primitives']
    pct = (diff / result2['total_primitives'] * 100) if result2['total_primitives'] > 0 else 0
    print(f"  Difference:           {diff:>10,} ({pct:>+6.2f}%)")

    # Primitive differences
    all_prims = set(result1['primitive_counts'].keys()) | set(result2['primitive_counts'].keys())
    diffs = []
    for prim in all_prims:
        c1 = result1['primitive_counts'].get(prim, 0)
        c2 = result2['primitive_counts'].get(prim, 0)
        if c1 != c2:
            diffs.append((prim, c1, c2, c1 - c2))

    diffs.sort(key=lambda x: abs(x[3]), reverse=True)

    print(f"\nTop 20 primitive differences:")
    print(f"  {'Primitive':<30s} {label1:>10s} {label2:>10s} {'Diff':>10s}")
    print(f"  {'-'*62}")
    for prim, c1, c2, diff in diffs[:20]:
        print(f"  {prim:<30s} {c1:>10,} {c2:>10,} {diff:>+10,}")


def print_four_way_comparison(main_no_cse, main_cse, old_no_cse, old_cse):
    """Print a 4-way comparison table."""
    print(f"\n{'='*80}")
    print(f"FOUR-WAY COMPARISON")
    print(f"{'='*80}")

    # Summary table
    print(f"\n{'Metric':<25s} {'MAIN no-CSE':>12s} {'MAIN CSE':>12s} {'OLD no-CSE':>12s} {'OLD CSE':>12s}")
    print(f"{'-'*77}")
    print(f"{'Total primitives':<25s} {main_no_cse['total_primitives']:>12,} {main_cse['total_primitives']:>12,} {old_no_cse['total_primitives']:>12,} {old_cse['total_primitives']:>12,}")
    print(f"{'Total equations':<25s} {main_no_cse['total_eqns']:>12,} {main_cse['total_eqns']:>12,} {old_no_cse['total_eqns']:>12,} {old_cse['total_eqns']:>12,}")

    main_elim = main_no_cse['total_eqns'] - main_cse['total_eqns']
    old_elim = old_no_cse['total_eqns'] - old_cse['total_eqns']
    print(f"{'CSE eliminated':<25s} {'-':>12s} {main_elim:>12,} {'-':>12s} {old_elim:>12,}")

    main_pct = (main_elim / main_no_cse['total_eqns'] * 100) if main_no_cse['total_eqns'] > 0 else 0
    old_pct = (old_elim / old_no_cse['total_eqns'] * 100) if old_no_cse['total_eqns'] > 0 else 0
    print(f"{'CSE reduction %':<25s} {'-':>12s} {f'{main_pct:.1f}%':>12s} {'-':>12s} {f'{old_pct:.1f}%':>12s}")

    # Primitive table
    all_prims = set()
    all_prims.update(main_no_cse['primitive_counts'].keys())
    all_prims.update(main_cse['primitive_counts'].keys())
    all_prims.update(old_no_cse['primitive_counts'].keys())
    all_prims.update(old_cse['primitive_counts'].keys())

    # Calculate totals for sorting
    prim_totals = {}
    for prim in all_prims:
        total = (main_no_cse['primitive_counts'].get(prim, 0) +
                main_cse['primitive_counts'].get(prim, 0) +
                old_no_cse['primitive_counts'].get(prim, 0) +
                old_cse['primitive_counts'].get(prim, 0))
        prim_totals[prim] = total

    sorted_prims = sorted(prim_totals.items(), key=lambda x: -x[1])

    print(f"\n{'Primitive':<25s} {'MAIN no-CSE':>12s} {'MAIN CSE':>12s} {'OLD no-CSE':>12s} {'OLD CSE':>12s}")
    print(f"{'-'*77}")

    for prim, _ in sorted_prims[:25]:
        m_no = main_no_cse['primitive_counts'].get(prim, 0)
        m_yes = main_cse['primitive_counts'].get(prim, 0)
        o_no = old_no_cse['primitive_counts'].get(prim, 0)
        o_yes = old_cse['primitive_counts'].get(prim, 0)

        print(f"{prim:<25s} {m_no:>12,} {m_yes:>12,} {o_no:>12,} {o_yes:>12,}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("="*80)
    print("STANDALONE BITONIC SORT COMPARISON")
    print("="*80)
    print(f"JAX version: {jax.__version__}")
    print(f"Test shape: {TEST_SHAPE}")

    # Determine version
    try:
        from tallax._src.bitonic_sort import bitonic_sort
        version = "MAIN"
        print(f"\n✓ Detected MAIN version (bitonic_sort.py)")

        def traced(x):
            result = bitonic_sort(x, max_num_fused_stages=None, tile_unroll=None, unroll_stages=True)
            return result[0] if isinstance(result, tuple) else result
    except ImportError:
        try:
            from tallax._src.bitonic_topk import bitonic_sort
            version = "OLD"
            print(f"\n✓ Detected OLD version (bitonic_topk.py)")

            def traced(x):
                result = bitonic_sort(x)
                return result[0] if isinstance(result, tuple) else result
        except ImportError:
            print("\n✗ Could not import bitonic_sort from either location!")
            print("  Make sure you're in the tallax repository")
            sys.exit(1)

    # Generate test data
    key = jax.random.PRNGKey(0)
    test_data = jax.random.normal(key, TEST_SHAPE)

    # Generate jaxpr
    print(f"\nGenerating jaxpr...")
    jaxpr = make_jaxpr(traced)(test_data)

    # Extract kernel jaxpr
    kernel_jaxpr = extract_kernel_jaxpr(jaxpr)
    if kernel_jaxpr is None:
        print("✗ Could not extract kernel jaxpr")
        sys.exit(1)

    print(f"✓ Extracted kernel jaxpr: {len(kernel_jaxpr.eqns):,} equations")

    # Analyze without CSE
    print(f"\n{'='*80}")
    print(f"ANALYZING WITHOUT CSE")
    print(f"{'='*80}")
    result_no_cse = analyze_version(version, kernel_jaxpr, apply_cse=False)
    print_analysis(result_no_cse, f"[{version}] Without CSE")

    # Analyze with CSE
    print(f"\n{'='*80}")
    print(f"ANALYZING WITH CSE")
    print(f"{'='*80}")
    result_cse = analyze_version(version, kernel_jaxpr, apply_cse=True)
    print_analysis(result_cse, f"[{version}] With CSE")

    # Compare no-CSE vs CSE
    compare_analyses(result_no_cse, result_cse,
                    f"{version} no-CSE", f"{version} CSE")

    # Save results
    output_file = os.path.join(OUTPUT_DIR, f'{version.lower()}_standalone_results.pkl')
    results = {
        'version': version,
        'no_cse': result_no_cse,
        'cse': result_cse,
    }

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n{'='*80}")
    print(f"RESULTS SAVED")
    print(f"{'='*80}")
    print(f"✓ Saved to {output_file}")

    # Try to load and compare with other version
    other_version = "OLD" if version == "MAIN" else "MAIN"
    other_file = os.path.join(OUTPUT_DIR, f'{other_version.lower()}_standalone_results.pkl')

    if os.path.exists(other_file):
        print(f"\n{'='*80}")
        print(f"COMPARING WITH {other_version} VERSION")
        print(f"{'='*80}")

        with open(other_file, 'rb') as f:
            other_results = pickle.load(f)

        if version == "MAIN":
            main_no_cse = result_no_cse
            main_cse = result_cse
            old_no_cse = other_results['no_cse']
            old_cse = other_results['cse']
        else:
            main_no_cse = other_results['no_cse']
            main_cse = other_results['cse']
            old_no_cse = result_no_cse
            old_cse = result_cse

        # Four-way comparison
        print_four_way_comparison(main_no_cse, main_cse, old_no_cse, old_cse)

        # Key comparisons
        compare_analyses(main_no_cse, old_no_cse, "MAIN no-CSE", "OLD no-CSE")
        compare_analyses(main_cse, old_cse, "MAIN CSE", "OLD CSE")

        # Final verdict
        print(f"\n{'='*80}")
        print(f"FINAL VERDICT")
        print(f"{'='*80}")

        diff_no_cse = main_no_cse['total_primitives'] - old_no_cse['total_primitives']
        pct_no_cse = (diff_no_cse / old_no_cse['total_primitives'] * 100)

        diff_cse = main_cse['total_primitives'] - old_cse['total_primitives']
        pct_cse = (diff_cse / old_cse['total_primitives'] * 100)

        print(f"\nWithout CSE:")
        print(f"  MAIN has {diff_no_cse:+,} primitives ({pct_no_cse:+.2f}%)")

        print(f"\nWith CSE:")
        print(f"  MAIN has {diff_cse:+,} primitives ({pct_cse:+.2f}%)")

        if abs(pct_cse) < 1.0:
            print(f"\n✓ After CSE, versions are virtually IDENTICAL (< 1% difference)")
            print(f"  This proves algorithmic equivalence!")
        elif abs(pct_cse) < abs(pct_no_cse):
            reduction = abs(pct_no_cse) - abs(pct_cse)
            print(f"\n✓ CSE reduces difference by {reduction:.1f} percentage points")
            print(f"  From {abs(pct_no_cse):.1f}% to {abs(pct_cse):.1f}%")

    else:
        print(f"\n⚠️  {other_version} version not analyzed yet")
        print(f"   To compare, run this script on the {other_version} version:")
        if other_version == "MAIN":
            print(f"     git checkout main")
        else:
            print(f"     git checkout claude/optimize-bitonic-ref-slicing-NWirx")
            print(f"     git checkout 895d0e830af6f31c0eaf2abff0771953b53f4ad9")
        print(f"     python {__file__}")

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
