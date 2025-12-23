"""
Test CSE behavior with jit primitives and verify it operates through them.
"""

import jax
import jax.numpy as jnp
from jax.extend.core import Jaxpr, Var, JaxprEqn, ClosedJaxpr, Literal
from collections import Counter
import hashlib


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
                # Count the jit if we can't recurse
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
                new_params[k] = ClosedJaxpr(cse_jaxpr(v.jaxpr, recurse_through_jit=recurse_through_jit), v.consts)
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


def test_simple_cse():
    """Test CSE with a simple function to verify it works through jit."""
    print("Testing CSE with simple function:")
    print("=" * 80)

    def simple_fn(x):
        a = x + 1
        b = x + 1  # Duplicate of a
        c = a * 2
        d = b * 2  # Should become a * 2 after CSE
        return c + d

    x = jnp.array(5.0)

    # Get jaxpr
    jaxpr_func = jax.make_jaxpr(simple_fn)(x)
    print("\nOriginal jaxpr:")
    print(jaxpr_func)

    # Count primitives (without recursing through jit)
    counts_no_recurse = count_primitives(jaxpr_func.jaxpr, recurse_through_jit=False)
    print(f"\nPrimitive counts (not recursing through jit): {dict(counts_no_recurse)}")

    # Count primitives (recursing through jit)
    counts_recurse = count_primitives(jaxpr_func.jaxpr, recurse_through_jit=True)
    print(f"Primitive counts (recursing through jit): {dict(counts_recurse)}")

    # Apply CSE
    cse_jaxpr_result = cse_jaxpr(jaxpr_func.jaxpr)
    print("\nCSE'd jaxpr:")
    # Print first few eqns to see structure
    for i, eqn in enumerate(cse_jaxpr_result.eqns[:5]):
        print(f"  {i}: {eqn}")

    # Count primitives after CSE
    cse_counts_recurse = count_primitives(cse_jaxpr_result, recurse_through_jit=True)
    print(f"\nCSE'd primitive counts: {dict(cse_counts_recurse)}")

    # Test fixpoint
    fixpoint_jaxpr, iterations = cse_until_fixpoint(jaxpr_func.jaxpr)
    fixpoint_counts = count_primitives(fixpoint_jaxpr, recurse_through_jit=True)
    print(f"\nFixpoint reached in {iterations} iterations")
    print(f"Fixpoint primitive counts: {dict(fixpoint_counts)}")

    # Verify correctness
    from jax.core import eval_jaxpr
    original_result = eval_jaxpr(jaxpr_func.jaxpr, jaxpr_func.consts, x)
    cse_result = eval_jaxpr(cse_jaxpr_result, jaxpr_func.consts, x)
    fixpoint_result = eval_jaxpr(fixpoint_jaxpr, jaxpr_func.consts, x)

    print(f"\nOriginal result: {original_result}")
    print(f"CSE result: {cse_result}")
    print(f"Fixpoint result: {fixpoint_result}")
    print(f"Match: {jnp.allclose(original_result, cse_result) and jnp.allclose(original_result, fixpoint_result)}")


if __name__ == "__main__":
    test_simple_cse()
