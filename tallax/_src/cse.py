"""Common Subexpression Elimination for JAX Jaxpr."""

import hashlib
from jax.extend.core import Jaxpr, Var, ClosedJaxpr, Literal


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
