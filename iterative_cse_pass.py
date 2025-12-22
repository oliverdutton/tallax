"""Iterative Common Subexpression Elimination (CSE) pass for JAX jaxpr.

Runs CSE repeatedly until fixpoint (no more changes).
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Any
import jax
from jax._src.core import Jaxpr, JaxprEqn, Var, Literal, ClosedJaxpr
from jax._src.util import safe_map
import hashlib

map = safe_map


def var_signature(v):
    """Create hashable signature for a variable or literal."""
    if isinstance(v, Var):
        # Use variable ID for identity-based comparison
        return ('Var', id(v), str(v.aval))
    elif isinstance(v, Literal):
        # For literals, hash the value
        try:
            return ('Literal', hash(v.val))
        except:
            # If unhashable, use string representation
            return ('Literal', str(v.val))
    else:
        return ('Other', str(type(v)), str(v))


def eqn_signature(eqn: JaxprEqn, var_mapping: Dict[Var, Var]) -> Tuple:
    """Create a hashable signature for an equation with current variable mappings.

    Two equations with the same signature compute the same value.
    """
    # Include primitive name
    prim_name = eqn.primitive.name

    # Include mapped input variables
    invars = tuple(
        var_signature(var_mapping.get(v, v))
        for v in eqn.invars
    )

    # Include parameters (only hashable ones)
    params = []
    for k, v in sorted(eqn.params.items()):
        # Skip nested jaxprs and complex structures
        if k in ['jaxpr', 'body_jaxpr', 'cond_jaxpr', 'branches', 'call_jaxpr']:
            continue
        try:
            hash(v)
            params.append((k, v))
        except TypeError:
            try:
                # Try converting to string
                params.append((k, str(v)))
            except:
                # Skip unhashable params that can't be stringified
                continue

    return (prim_name, invars, tuple(params))


def count_jaxpr_equations(jaxpr: Jaxpr) -> int:
    """Count total equations including nested jaxprs."""
    count = len(jaxpr.eqns)

    for eqn in jaxpr.eqns:
        for param_name in ['jaxpr', 'body_jaxpr', 'cond_jaxpr']:
            if param_name in eqn.params:
                nested = eqn.params[param_name]
                if hasattr(nested, 'jaxpr'):
                    count += count_jaxpr_equations(nested.jaxpr)
                elif isinstance(nested, Jaxpr):
                    count += count_jaxpr_equations(nested)

    return count


def apply_cse_single_pass(jaxpr: Jaxpr, recursive: bool = True) -> Tuple[Jaxpr, int]:
    """Apply one pass of CSE to a jaxpr.

    Returns:
        (optimized_jaxpr, num_eliminations)
    """
    # Map from equation signature to its output variable
    eqn_cache: Dict[Tuple, Var] = {}

    # Map from old output var to new output var (for deduplication)
    var_mapping: Dict[Var, Var] = {}

    # Track input variables - these can't be eliminated
    input_vars = set(jaxpr.invars)

    # New equations list
    new_eqns: List[JaxprEqn] = []
    eliminations = 0

    for eqn in jaxpr.eqns:
        # Process nested jaxprs recursively
        processed_eqn = eqn
        nested_eliminations = 0

        if recursive:
            for param_name in ['jaxpr', 'body_jaxpr', 'cond_jaxpr']:
                if param_name in eqn.params:
                    nested_jaxpr = eqn.params[param_name]
                    if hasattr(nested_jaxpr, 'jaxpr'):
                        # ClosedJaxpr
                        opt_nested, nested_elims = apply_cse_to_closed_jaxpr(nested_jaxpr)
                        processed_eqn = processed_eqn.replace(
                            params={**processed_eqn.params, param_name: opt_nested}
                        )
                        nested_eliminations += nested_elims
                    elif isinstance(nested_jaxpr, Jaxpr):
                        # Plain Jaxpr
                        opt_nested, nested_elims = apply_cse_single_pass(nested_jaxpr, recursive=True)
                        processed_eqn = processed_eqn.replace(
                            params={**processed_eqn.params, param_name: opt_nested}
                        )
                        nested_eliminations += nested_elims

        # Map input variables to their canonical versions
        mapped_invars = [
            var_mapping.get(v, v) if isinstance(v, Var) else v
            for v in processed_eqn.invars
        ]

        # Create equation with mapped invars
        mapped_eqn = processed_eqn.replace(invars=mapped_invars)

        # Check if this is a side-effecting primitive (don't eliminate these)
        is_pure = not eqn.effects
        has_single_output = len(processed_eqn.outvars) == 1

        if is_pure and has_single_output:
            # Compute signature with mapped inputs
            try:
                sig = eqn_signature(mapped_eqn, var_mapping)

                # Check if we've seen this computation before
                if sig in eqn_cache:
                    # We've computed this before, reuse the result
                    old_outvar = processed_eqn.outvars[0]
                    cached_outvar = eqn_cache[sig]

                    # Don't eliminate if the output is an input variable
                    if old_outvar not in input_vars:
                        var_mapping[old_outvar] = cached_outvar
                        eliminations += 1
                        # Don't add this equation to new_eqns
                        continue

                # New computation, add to cache
                eqn_cache[sig] = processed_eqn.outvars[0]
            except:
                # If signature creation fails, just keep the equation
                pass

        # Add equation to new list
        new_eqns.append(mapped_eqn)
        eliminations += nested_eliminations

    # Update outvars with mapping
    new_outvars = [
        var_mapping.get(v, v) if isinstance(v, Var) else v
        for v in jaxpr.outvars
    ]

    return jaxpr.replace(eqns=new_eqns, outvars=new_outvars), eliminations


def apply_cse_to_closed_jaxpr(closed_jaxpr: ClosedJaxpr) -> Tuple[ClosedJaxpr, int]:
    """Apply CSE to a ClosedJaxpr.

    Returns:
        (optimized_jaxpr, num_eliminations)
    """
    optimized_jaxpr, eliminations = apply_cse_single_pass(closed_jaxpr.jaxpr, recursive=True)
    return closed_jaxpr.replace(jaxpr=optimized_jaxpr), eliminations


def apply_iterative_cse(jaxpr_or_closed, max_iterations=100, verbose=True):
    """Apply CSE iteratively until fixpoint.

    Args:
        jaxpr_or_closed: Jaxpr or ClosedJaxpr to optimize
        max_iterations: Maximum number of CSE passes
        verbose: Print progress

    Returns:
        (optimized_jaxpr, total_eliminations, num_iterations)
    """
    is_closed = hasattr(jaxpr_or_closed, 'jaxpr')
    current = jaxpr_or_closed
    total_eliminations = 0
    iteration = 0

    initial_eqn_count = count_jaxpr_equations(
        current.jaxpr if is_closed else current
    )

    if verbose:
        print(f"Initial equation count: {initial_eqn_count}")
        print("Starting iterative CSE...")

    while iteration < max_iterations:
        iteration += 1

        # Apply one pass of CSE
        if is_closed:
            optimized, eliminations = apply_cse_to_closed_jaxpr(current)
        else:
            optimized, eliminations = apply_cse_single_pass(current, recursive=True)

        if verbose:
            current_count = count_jaxpr_equations(
                optimized.jaxpr if is_closed else optimized
            )
            print(f"  Iteration {iteration}: eliminated {eliminations} operations "
                  f"(total equations: {current_count})")

        # Check for fixpoint
        if eliminations == 0:
            if verbose:
                print(f"Reached fixpoint after {iteration} iterations")
            break

        total_eliminations += eliminations
        current = optimized

    final_eqn_count = count_jaxpr_equations(
        current.jaxpr if is_closed else current
    )

    if verbose:
        print(f"\nCSE Results:")
        print(f"  Total iterations: {iteration}")
        print(f"  Total eliminations: {total_eliminations}")
        print(f"  Equations: {initial_eqn_count} → {final_eqn_count}")
        print(f"  Reduction: {(1 - final_eqn_count/initial_eqn_count)*100:.1f}%")

    return current, total_eliminations, iteration


if __name__ == "__main__":
    import jax.numpy as jnp
    from tallax._src.sort import sort

    print("="*80)
    print("Testing Iterative CSE on Bitonic Sort")
    print("="*80)

    # Create test input
    shape = (8, 1024)
    key = jax.random.PRNGKey(42)
    arr = jax.random.normal(key, shape, dtype=jnp.float32)

    # Create the jaxpr
    print("\nGenerating jaxpr for bitonic sort...")

    def sort_func(x):
        return sort([x], num_keys=1, descending=False, is_stable=False, return_argsort=False)

    original_jaxpr = jax.make_jaxpr(sort_func)(arr)

    print(f"Original jaxpr equations: {len(original_jaxpr.jaxpr.eqns)}")
    print(f"Total equations (including nested): {count_jaxpr_equations(original_jaxpr.jaxpr)}")

    # Apply iterative CSE
    print("\n" + "="*80)
    optimized_jaxpr, total_elims, num_iters = apply_iterative_cse(
        original_jaxpr,
        max_iterations=100,
        verbose=True
    )

    # Save results
    with open('/home/user/tallax/cse_optimized_jaxpr.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("ORIGINAL JAXPR\n")
        f.write("="*80 + "\n")
        f.write(str(original_jaxpr))
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("CSE-OPTIMIZED JAXPR\n")
        f.write("="*80 + "\n")
        f.write(str(optimized_jaxpr))

    print(f"\nFull jaxprs saved to: /home/user/tallax/cse_optimized_jaxpr.txt")
