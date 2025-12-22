"""Enhanced CSE pass with better constant handling and value propagation."""

from collections import defaultdict
from typing import Dict, List, Tuple, Set
import jax
from jax._src.core import Jaxpr, JaxprEqn, Var, Literal, ClosedJaxpr
from jax._src.util import safe_map
import hashlib

map = safe_map


def value_signature(v):
    """Create signature that tracks both identity AND value for constants."""
    if isinstance(v, Literal):
        # For literals, hash the actual value content
        try:
            # Handle arrays
            if hasattr(v.val, 'tobytes'):
                return ('Literal_val', hashlib.md5(v.val.tobytes()).hexdigest(), str(v.aval))
            # Handle scalars
            return ('Literal_val', hash(v.val), str(v.aval))
        except:
            return ('Literal_str', str(v.val), str(v.aval))
    elif isinstance(v, Var):
        return ('Var', id(v), str(v.aval))
    else:
        return ('Other', str(type(v)), str(v))


def eqn_value_signature(eqn: JaxprEqn, value_map: Dict) -> Tuple:
    """Create signature based on COMPUTED VALUES, not variable identity.

    This allows CSE to recognize when the same computation is performed
    with the same input values, even if variable IDs differ.
    """
    prim_name = eqn.primitive.name

    # Get value signatures for inputs
    invars = tuple(
        value_map.get(id(v), value_signature(v))
        for v in eqn.invars
    )

    # Include primitive parameters
    params = []
    for k, v in sorted(eqn.params.items()):
        if k in ['jaxpr', 'body_jaxpr', 'cond_jaxpr', 'branches', 'call_jaxpr']:
            continue
        try:
            # For array params, hash content
            if hasattr(v, 'tobytes'):
                params.append((k, hashlib.md5(v.tobytes()).hexdigest()))
            else:
                hash(v)
                params.append((k, v))
        except:
            try:
                params.append((k, str(v)))
            except:
                continue

    return (prim_name, invars, tuple(params))


def count_jaxpr_equations(jaxpr: Jaxpr) -> int:
    """Count total equations including nested."""
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


def enhanced_cse_pass(jaxpr: Jaxpr, recursive: bool = True) -> Tuple[Jaxpr, int]:
    """Enhanced CSE with value-based deduplication.

    Key improvements:
    1. Tracks computed values, not just variable IDs
    2. Recognizes that broadcast_in_dim of same constant creates same value
    3. Identifies iota operations with same parameters as duplicates
    """
    # Map from equation value signature to output variable
    value_cache: Dict[Tuple, Var] = {}

    # Map from variable ID to its value signature (for propagation)
    value_map: Dict[int, Tuple] = {}

    # Map from old var to new var (for replacement)
    var_mapping: Dict[Var, Var] = {}

    # Track input variables
    input_vars = set(jaxpr.invars)

    new_eqns: List[JaxprEqn] = []
    eliminations = 0

    for eqn in jaxpr.eqns:
        # Process nested jaxprs
        processed_eqn = eqn
        nested_elims = 0

        if recursive:
            for param_name in ['jaxpr', 'body_jaxpr', 'cond_jaxpr']:
                if param_name in eqn.params:
                    nested_jaxpr = eqn.params[param_name]
                    if hasattr(nested_jaxpr, 'jaxpr'):
                        opt_nested, nested_elims_count = enhanced_cse_to_closed_jaxpr(nested_jaxpr)
                        processed_eqn = processed_eqn.replace(
                            params={**processed_eqn.params, param_name: opt_nested}
                        )
                        nested_elims += nested_elims_count
                    elif isinstance(nested_jaxpr, Jaxpr):
                        opt_nested, nested_elims_count = enhanced_cse_pass(nested_jaxpr, recursive=True)
                        processed_eqn = processed_eqn.replace(
                            params={**processed_eqn.params, param_name: opt_nested}
                        )
                        nested_elims += nested_elims_count

        # Map input variables
        mapped_invars = [
            var_mapping.get(v, v) if isinstance(v, Var) else v
            for v in processed_eqn.invars
        ]

        mapped_eqn = processed_eqn.replace(invars=mapped_invars)

        # Check if pure and single-output
        is_pure = not eqn.effects
        has_single_output = len(processed_eqn.outvars) == 1

        if is_pure and has_single_output:
            try:
                # Compute value-based signature
                val_sig = eqn_value_signature(mapped_eqn, value_map)

                if val_sig in value_cache:
                    # Found duplicate computation!
                    old_outvar = processed_eqn.outvars[0]
                    cached_outvar = value_cache[val_sig]

                    if old_outvar not in input_vars:
                        var_mapping[old_outvar] = cached_outvar
                        # Propagate value signature
                        value_map[id(old_outvar)] = value_map.get(id(cached_outvar),
                                                                  value_signature(cached_outvar))
                        eliminations += 1
                        continue

                # New computation - cache it
                outvar = processed_eqn.outvars[0]
                value_cache[val_sig] = outvar

                # Store value signature for this output
                # This allows downstream ops to recognize same values
                value_map[id(outvar)] = val_sig

            except Exception as e:
                # If signature fails, just keep the equation
                pass

        new_eqns.append(mapped_eqn)
        eliminations += nested_elims

    # Update outvars
    new_outvars = [
        var_mapping.get(v, v) if isinstance(v, Var) else v
        for v in jaxpr.outvars
    ]

    return jaxpr.replace(eqns=new_eqns, outvars=new_outvars), eliminations


def enhanced_cse_to_closed_jaxpr(closed_jaxpr: ClosedJaxpr) -> Tuple[ClosedJaxpr, int]:
    """Apply enhanced CSE to ClosedJaxpr."""
    optimized_jaxpr, eliminations = enhanced_cse_pass(closed_jaxpr.jaxpr, recursive=True)
    return closed_jaxpr.replace(jaxpr=optimized_jaxpr), eliminations


def apply_enhanced_iterative_cse(jaxpr_or_closed, max_iterations=100, verbose=True):
    """Apply enhanced CSE iteratively until fixpoint."""
    is_closed = hasattr(jaxpr_or_closed, 'jaxpr')
    current = jaxpr_or_closed
    total_eliminations = 0
    iteration = 0

    initial_eqn_count = count_jaxpr_equations(
        current.jaxpr if is_closed else current
    )

    if verbose:
        print(f"Initial equation count: {initial_eqn_count}")
        print("Starting enhanced iterative CSE...")

    while iteration < max_iterations:
        iteration += 1

        if is_closed:
            optimized, eliminations = enhanced_cse_to_closed_jaxpr(current)
        else:
            optimized, eliminations = enhanced_cse_pass(current, recursive=True)

        if verbose:
            current_count = count_jaxpr_equations(
                optimized.jaxpr if is_closed else optimized
            )
            print(f"  Iteration {iteration}: eliminated {eliminations} operations "
                  f"(total equations: {current_count})")

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
        print(f"\nEnhanced CSE Results:")
        print(f"  Total iterations: {iteration}")
        print(f"  Total eliminations: {total_eliminations}")
        print(f"  Equations: {initial_eqn_count} → {final_eqn_count}")
        if initial_eqn_count > 0:
            print(f"  Reduction: {(1 - final_eqn_count/initial_eqn_count)*100:.1f}%")

    return current, total_eliminations, iteration


if __name__ == "__main__":
    import jax.numpy as jnp
    from tallax._src.sort import sort

    print("="*80)
    print("Testing Enhanced CSE on Bitonic Sort")
    print("="*80)

    # Create test input
    shape = (8, 1024)
    key = jax.random.PRNGKey(42)
    arr = jax.random.normal(key, shape, dtype=jnp.float32)

    def sort_func(x):
        return sort([x], num_keys=1, descending=False, is_stable=False, return_argsort=False)

    print("\nGenerating jaxpr...")
    original_jaxpr = jax.make_jaxpr(sort_func)(arr)

    # Apply enhanced CSE
    print("\n" + "="*80)
    optimized_jaxpr, total_elims, num_iters = apply_enhanced_iterative_cse(
        original_jaxpr,
        max_iterations=100,
        verbose=True
    )

    # Analyze iota count
    import re
    jaxpr_str = str(optimized_jaxpr)
    iota_count = len(re.findall(r'\w+:\w+\[\d+,\d+\]\s*=\s*iota\[', jaxpr_str))

    print(f"\n" + "="*80)
    print(f"IOTA OPERATIONS")
    print(f"="*80)
    print(f"After enhanced CSE: {iota_count} iota operations")
    print(f"Expected (with proper hoisting): ~2-10")
    print(f"Status: {'✓ GOOD' if iota_count < 20 else '✗ NEEDS MORE WORK'}")
