
import functools
import pytest

import jax
import jax.numpy as jnp

from tallax import tax
from tallax.utils import is_cpu_platform

@jax.jit
def exact_match(xs, ys):
  """Check if two pytrees match exactly (including NaN positions)."""
  def _all(equality_op):
    return jnp.array(jax.tree.leaves(
        jax.tree.map(lambda x, y: equality_op(x, y).all(), xs, ys)
    )).all()

  nans_match = _all(lambda x, y: jnp.isnan(x) == jnp.isnan(y))
  non_nans_match = _all(lambda x, y: jnp.where(jnp.isnan(x), True, x == y))
  return nans_match & non_nans_match


@functools.partial(
    jax.jit,
    static_argnames=('num_vmem_substages', 'descending', 'return_argsort',
                     'is_stable', 'num_keys', 'block_token', 'interpret')
)
def _equiv_xla_based_sort(
    operand,
    num_keys: int,
    is_stable: bool = False,
    return_argsort: bool = False,
    descending: bool = False,
    num_vmem_substages: int | None = None,
    block_token: int | None = None,
    interpret: bool | None = None,
) -> tuple[jax.Array, ...]:
  """Reference implementation using XLA sort for correctness testing."""
  del num_vmem_substages, block_token, interpret
  operands = jax.tree.leaves(operand)

  if return_argsort:
    operands.append(
        jax.lax.broadcasted_iota(jnp.int32, operands[0].shape, 1)
    )
  if descending and is_stable:
    operands.insert(
        num_keys,
        -jax.lax.broadcasted_iota(jnp.int32, operands[0].shape, 1)
    )
    num_keys += 1

  outs = jax.lax.sort(operands, num_keys=num_keys, is_stable=is_stable)

  if descending and is_stable:
    outs = list(outs)
    outs.pop(num_keys - 1)
  if descending:
    outs = tuple(x[..., ::-1] for x in outs)

  return tuple(outs)


def verify_sort(
    operand,
    num_keys: int,
    block_token: int | None = None,
    return_argsort: bool = False,
    descending: bool = False,
    is_stable: bool = False,
    print_outputs: bool = False,
    interpret: bool | None = None,
):
  """Validate sort against XLA reference implementation."""
  if interpret is None:
    interpret = is_cpu_platform()

  kwargs = dict(
      block_token=block_token,
      return_argsort=return_argsort,
      descending=descending,
      num_keys=num_keys,
      is_stable=is_stable,
      interpret=interpret
  )
  out_pallas = tax.sort(operand, **kwargs)

  if is_stable:
    # Exact match required for stable sort
    kwargs_for_xla = kwargs.copy()
    out_xla = _equiv_xla_based_sort(operand, **kwargs_for_xla)
    valid = exact_match(out_pallas, out_xla)

    if not valid:
      m = jnp.zeros(out_xla[0].shape, dtype=bool)
      for ox, op in zip(out_xla, out_pallas):
        m |= ~((ox == op) | (jnp.isnan(ox) & jnp.isnan(op)))
      debug_msg = []
      for ox, op in zip(out_xla, out_pallas):
        debug_msg.append(f'xla {ox[m]}\npallas {op[m]}')
      debug_output = '\n'.join(debug_msg)
      pytest.fail(f"Pallas output does not match XLA output for stable sort:\n{debug_output}")

    assert valid, "Pallas output does not match XLA output for stable sort"

  else:
    # Check output is valid permutation with correct relative order
    out_pallas_stable_sorted = _equiv_xla_based_sort(
        out_pallas,
        num_keys=num_keys,
        is_stable=True,
        descending=descending,
        interpret=interpret,
    )
    valid = exact_match(out_pallas, out_pallas_stable_sorted)
    if not valid:
      m = jnp.zeros(out_pallas_stable_sorted[0].shape, dtype=bool)
      for ox, op in zip(out_pallas_stable_sorted, out_pallas):
        m |= ~((ox == op) | (jnp.isnan(ox) & jnp.isnan(op)))
      debug_msg = []
      for ox, op in zip(out_pallas_stable_sorted, out_pallas):
        debug_msg.append(f'sorted {ox[m]}\npallas {op[m]}')
      debug_output = '\n'.join(debug_msg)
      pytest.fail(f"Pallas output is not sorted:\n{debug_output}")

    assert valid, "out_pallas must be sorted (verified by re-sorting stably)"

    narrs = len(out_pallas)
    kwargs_for_xla = kwargs.copy()
    operands_fully_sorted = _equiv_xla_based_sort(
        operand, **{**kwargs_for_xla, 'num_keys': narrs}
    )
    out_pallas_fully_sorted = _equiv_xla_based_sort(
        out_pallas, **{**kwargs_for_xla, 'num_keys': narrs, 'return_argsort': False}
    )
    valid_permute = exact_match(operands_fully_sorted, out_pallas_fully_sorted)
    assert valid_permute, "out_pallas is not a valid permutation of input"
    valid &= valid_permute

  if print_outputs:
    o_pallas, o_xla = _equiv_xla_based_sort(operand, **kwargs)
    print(f'Pallas: {o_pallas}\nXLA: {o_xla}')

@pytest.mark.parametrize("is_stable", [False, True])
@pytest.mark.parametrize("return_argsort", [False, True])
@pytest.mark.parametrize("descending", [False, True])
def test_sort(is_stable, return_argsort, descending):
  shape = (8, 16) if is_cpu_platform() else (8, 128)
  operands = [jax.random.randint(jax.random.key(0), shape, 0, 100, jnp.int32)]
  verify_sort(
      operands,
      num_keys=1,
      is_stable=is_stable,
      return_argsort=return_argsort,
      descending=descending
  )
