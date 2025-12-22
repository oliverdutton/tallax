#!/usr/bin/env python3
"""Script to analyze Pallas jaxpr for rematerialization issues."""

import jax
import jax.numpy as jnp
from tallax._src.sort import sort
from collections import Counter
import sys


def get_pallas_jaxpr(func, *args):
    """Extract the jaxpr from a function."""
    return jax.make_jaxpr(func)(*args)


def analyze_jaxpr_for_rematerialization(jaxpr):
    """Analyze a jaxpr for potential rematerialization.

    Rematerialization occurs when the same computation is performed multiple times
    instead of computing once and reusing the result.
    """
    # Count operations by their equation string representation
    eqn_strings = []
    eqn_details = []

    def process_jaxpr_recursive(j, prefix=""):
        nonlocal eqn_strings, eqn_details
        for i, eqn in enumerate(j.eqns):
            # Create a string representation of the equation for comparison
            eqn_str = f"{eqn.primitive.name}({','.join(str(v) for v in eqn.invars)})"
            eqn_strings.append(eqn_str)
            eqn_details.append({
                'prefix': prefix,
                'index': i,
                'primitive': eqn.primitive.name,
                'invars': [str(v) for v in eqn.invars],
                'outvars': [str(v) for v in eqn.outvars],
                'params': eqn.params,
                'eqn_str': eqn_str
            })

            # Process nested jaxprs (for pallas_call, while, cond, etc.)
            if 'jaxpr' in eqn.params:
                nested_jaxpr = eqn.params['jaxpr']
                if hasattr(nested_jaxpr, 'jaxpr'):
                    nested_jaxpr = nested_jaxpr.jaxpr
                process_jaxpr_recursive(nested_jaxpr, prefix=f"{prefix}>{eqn.primitive.name}")

            if 'body_jaxpr' in eqn.params:
                body_jaxpr = eqn.params['body_jaxpr']
                if hasattr(body_jaxpr, 'jaxpr'):
                    body_jaxpr = body_jaxpr.jaxpr
                process_jaxpr_recursive(body_jaxpr, prefix=f"{prefix}>{eqn.primitive.name}_body")

    process_jaxpr_recursive(jaxpr.jaxpr)

    # Count duplicates
    eqn_counts = Counter(eqn_strings)
    duplicates = {k: v for k, v in eqn_counts.items() if v > 1}

    return {
        'total_equations': len(eqn_strings),
        'unique_equations': len(set(eqn_strings)),
        'duplicate_count': sum(v - 1 for v in duplicates.values()),
        'duplicates': duplicates,
        'all_equations': eqn_details,
        'jaxpr': jaxpr
    }


def print_analysis(analysis, verbose=False):
    """Print analysis results."""
    print(f"\n{'='*80}")
    print("JAXPR Rematerialization Analysis")
    print(f"{'='*80}")
    print(f"Total equations: {analysis['total_equations']}")
    print(f"Unique equations: {analysis['unique_equations']}")
    print(f"Duplicate operations: {analysis['duplicate_count']}")
    print(f"Potential rematerializations: {len(analysis['duplicates'])}")

    if analysis['duplicates']:
        print(f"\n{'='*80}")
        print("Most Duplicated Operations:")
        print(f"{'='*80}")
        sorted_dups = sorted(analysis['duplicates'].items(), key=lambda x: x[1], reverse=True)
        for eqn_str, count in sorted_dups[:20]:  # Show top 20
            print(f"  [{count}x] {eqn_str[:100]}")

    if verbose:
        print(f"\n{'='*80}")
        print("Full Jaxpr:")
        print(f"{'='*80}")
        print(analysis['jaxpr'])


def find_pallas_call_jaxpr(jaxpr):
    """Find and extract pallas_call jaxprs."""
    pallas_jaxprs = []

    def search_jaxpr(j, depth=0):
        for eqn in j.eqns:
            if 'pallas_call' in eqn.primitive.name:
                if 'jaxpr' in eqn.params:
                    nested = eqn.params['jaxpr']
                    if hasattr(nested, 'jaxpr'):
                        nested = nested.jaxpr
                    pallas_jaxprs.append({
                        'depth': depth,
                        'jaxpr': nested,
                        'params': eqn.params
                    })

            # Recurse into nested jaxprs
            for param_name in ['jaxpr', 'body_jaxpr', 'cond_jaxpr']:
                if param_name in eqn.params:
                    nested = eqn.params[param_name]
                    if hasattr(nested, 'jaxpr'):
                        nested = nested.jaxpr
                    search_jaxpr(nested, depth + 1)

    search_jaxpr(jaxpr.jaxpr)
    return pallas_jaxprs


def main():
    print("Analyzing bitonic sort for rematerialization...")
    print(f"Configuration: shape=(8, 1024), pipeline_stages=4")

    # Create test input
    shape = (8, 1024)
    key = jax.random.PRNGKey(42)
    arr = jax.random.normal(key, shape, dtype=jnp.float32)

    # Create the jaxpr
    print("\nGenerating jaxpr...")

    def sort_func(x):
        return sort([x], num_keys=1, descending=False, is_stable=False, return_argsort=False)

    jaxpr = get_pallas_jaxpr(sort_func, arr)

    print(f"\nMain jaxpr has {len(jaxpr.jaxpr.eqns)} equations")

    # Find pallas_call jaxprs
    pallas_jaxprs = find_pallas_call_jaxpr(jaxpr)
    print(f"Found {len(pallas_jaxprs)} pallas_call operations")

    # Analyze main jaxpr
    print("\n" + "="*80)
    print("MAIN JAXPR ANALYSIS")
    analysis = analyze_jaxpr_for_rematerialization(jaxpr)
    print_analysis(analysis, verbose=False)

    # Analyze each pallas_call jaxpr
    for i, pallas_info in enumerate(pallas_jaxprs):
        print(f"\n{'='*80}")
        print(f"PALLAS_CALL #{i+1} (depth={pallas_info['depth']}) ANALYSIS")

        # Create a simple wrapper to analyze
        class FakeJaxpr:
            def __init__(self, j):
                self.jaxpr = j

        fake_jaxpr = FakeJaxpr(pallas_info['jaxpr'])
        analysis = analyze_jaxpr_for_rematerialization(fake_jaxpr)
        print_analysis(analysis, verbose=False)

        print(f"\nJaxpr length (equation count): {len(pallas_info['jaxpr'].eqns)}")

    # Save full jaxpr to file
    with open('/home/user/tallax/jaxpr_output.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("FULL JAXPR\n")
        f.write("="*80 + "\n")
        f.write(str(jaxpr))
        f.write("\n\n")

        for i, pallas_info in enumerate(pallas_jaxprs):
            f.write(f"\n{'='*80}\n")
            f.write(f"PALLAS_CALL #{i+1}\n")
            f.write(f"{'='*80}\n")
            f.write(str(pallas_info['jaxpr']))
            f.write("\n\n")

    print(f"\n{'='*80}")
    print(f"Full jaxpr saved to: /home/user/tallax/jaxpr_output.txt")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
