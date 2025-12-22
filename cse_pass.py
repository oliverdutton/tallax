"""Common Subexpression Elimination (CSE) pass for JAX jaxpr."""

from collections import defaultdict
from typing import Dict, List, Tuple
import jax
from jax._src.core import Jaxpr, JaxprEqn, Var, Literal, ClosedJaxpr
from jax._src.util import safe_map

map = safe_map


def eqn_signature(eqn: JaxprEqn) -> Tuple:
    """Create a hashable signature for an equation.

    Two equations with the same signature compute the same value.
    """
    # Include primitive name
    prim_name = eqn.primitive.name

    # Include input variables (by their identity, not value)
    invars = tuple(
        (type(v).__name__, id(v) if isinstance(v, Var) else v.val)
        for v in eqn.invars
    )

    # Include parameters (only hashable ones)
    # Skip nested jaxprs for now as they're complex to hash
    params = []
    for k, v in sorted(eqn.params.items()):
        if k in ['jaxpr', 'body_jaxpr', 'cond_jaxpr', 'branches']:
            continue  # Skip nested jaxprs
        try:
            # Try to hash the parameter
            hash(v)
            params.append((k, v))
        except TypeError:
            # If not hashable, convert to string
            params.append((k, str(v)))

    return (prim_name, invars, tuple(params))


def apply_cse_to_jaxpr(jaxpr: Jaxpr, recursive: bool = True) -> Jaxpr:
    """Apply Common Subexpression Elimination to a jaxpr.

    Args:
        jaxpr: The jaxpr to optimize
        recursive: Whether to recursively apply CSE to nested jaxprs

    Returns:
        Optimized jaxpr with duplicates eliminated
    """
    # Map from equation signature to its output variable
    eqn_cache: Dict[Tuple, Var] = {}

    # Map from old output var to new output var (for deduplication)
    var_mapping: Dict[Var, Var] = {}

    # New equations list
    new_eqns: List[JaxprEqn] = []

    for eqn in jaxpr.eqns:
        # Process nested jaxprs if recursive
        processed_eqn = eqn
        if recursive and 'jaxpr' in eqn.params:
            nested_jaxpr = eqn.params['jaxpr']
            if hasattr(nested_jaxpr, 'jaxpr'):
                # ClosedJaxpr
                optimized_nested = apply_cse_to_closed_jaxpr(nested_jaxpr)
                processed_eqn = eqn.replace(
                    params={**eqn.params, 'jaxpr': optimized_nested}
                )
            else:
                # Plain Jaxpr
                optimized_nested = apply_cse_to_jaxpr(nested_jaxpr, recursive=True)
                processed_eqn = eqn.replace(
                    params={**eqn.params, 'jaxpr': optimized_nested}
                )

        if recursive and 'body_jaxpr' in eqn.params:
            body_jaxpr = eqn.params['body_jaxpr']
            if hasattr(body_jaxpr, 'jaxpr'):
                optimized_body = apply_cse_to_closed_jaxpr(body_jaxpr)
                processed_eqn = eqn.replace(
                    params={**eqn.params, 'body_jaxpr': optimized_body}
                )
            else:
                optimized_body = apply_cse_to_jaxpr(body_jaxpr, recursive=True)
                processed_eqn = eqn.replace(
                    params={**eqn.params, 'body_jaxpr': optimized_body}
                )

        # Map input variables to their canonical versions
        mapped_invars = [
            var_mapping.get(v, v) if isinstance(v, Var) else v
            for v in processed_eqn.invars
        ]

        # Create equation with mapped invars
        mapped_eqn = processed_eqn.replace(invars=mapped_invars)

        # Compute signature with mapped inputs
        sig = eqn_signature(mapped_eqn)

        # Check if we've seen this computation before
        if sig in eqn_cache and len(processed_eqn.outvars) == 1:
            # We've computed this before, reuse the result
            old_outvar = processed_eqn.outvars[0]
            cached_outvar = eqn_cache[sig]
            var_mapping[old_outvar] = cached_outvar
            # Don't add this equation to new_eqns
        else:
            # New computation, add to cache and equations
            if len(processed_eqn.outvars) == 1:
                eqn_cache[sig] = processed_eqn.outvars[0]
            new_eqns.append(mapped_eqn)

    # Update outvars with mapping
    new_outvars = [
        var_mapping.get(v, v) if isinstance(v, Var) else v
        for v in jaxpr.outvars
    ]

    return jaxpr.replace(eqns=new_eqns, outvars=new_outvars)


def apply_cse_to_closed_jaxpr(closed_jaxpr: ClosedJaxpr) -> ClosedJaxpr:
    """Apply CSE to a ClosedJaxpr."""
    optimized_jaxpr = apply_cse_to_jaxpr(closed_jaxpr.jaxpr, recursive=True)
    return closed_jaxpr.replace(jaxpr=optimized_jaxpr)


def optimize_jaxpr_cse(jaxpr_or_closed):
    """Public API for applying CSE optimization."""
    if hasattr(jaxpr_or_closed, 'jaxpr'):
        # ClosedJaxpr
        return apply_cse_to_closed_jaxpr(jaxpr_or_closed)
    else:
        # Plain Jaxpr
        return apply_cse_to_jaxpr(jaxpr_or_closed, recursive=True)


def create_cse_optimized_function(func):
    """Decorator to apply CSE to a function's jaxpr.

    Note: This works at the jaxpr level, so it won't affect XLA compilation.
    For production use, you'd need to integrate this into the compiler pipeline.
    """
    def wrapper(*args, **kwargs):
        # Create jaxpr
        jaxpr = jax.make_jaxpr(func)(*args, **kwargs)

        # Optimize
        optimized_jaxpr = optimize_jaxpr_cse(jaxpr)

        # Evaluate the optimized jaxpr
        from jax.core import eval_jaxpr
        result = eval_jaxpr(optimized_jaxpr.jaxpr, optimized_jaxpr.consts, *args)

        return result

    return wrapper


if __name__ == "__main__":
    # Simple test
    import jax.numpy as jnp

    def test_func(x):
        a = x + 1
        b = x + 1  # Duplicate computation
        c = a * 2
        d = b * 2  # Should reuse a's result
        return c + d

    x = jnp.array([1.0, 2.0, 3.0])

    # Original jaxpr
    original_jaxpr = jax.make_jaxpr(test_func)(x)
    print("Original jaxpr equations:", len(original_jaxpr.jaxpr.eqns))

    # Optimized jaxpr
    optimized_jaxpr = optimize_jaxpr_cse(original_jaxpr)
    print("Optimized jaxpr equations:", len(optimized_jaxpr.jaxpr.eqns))

    print("\nOriginal:")
    print(original_jaxpr)
    print("\nOptimized:")
    print(optimized_jaxpr)
