
import functools

import jax
import jax.numpy as jnp

from tallax import lax_sort_pallas

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
                     'is_stable', 'num_keys', 'block_token')
)
def _equiv_xla_based_sort(
    operand,
    num_keys: int = 1,
    is_stable: bool = False,
    return_argsort: bool = False,
    descending: bool = False,
    num_vmem_substages: int | None = None,
    block_token: int | None = None,
) -> tuple[jax.Array, ...]:
  """Reference implementation using XLA sort for correctness testing."""
  del num_vmem_substages, block_token
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


def check_lax_sort_pallas(
    operand,
    block_token: int | None = None,
    return_argsort: bool = False,
    descending: bool = False,
    num_keys: int | None = None,
    is_stable: bool = False,
    print_outputs: bool = False,
):
  """Validate lax_sort_pallas against XLA reference implementation."""
  kwargs = dict(
      block_token=block_token,
      return_argsort=return_argsort,
      descending=descending,
      num_keys=num_keys,
      is_stable=is_stable
  )
  out_pallas = lax_sort_pallas(operand, **kwargs)

  if is_stable:
    # Exact match required for stable sort
    out_xla = _equiv_xla_based_sort(operand, **kwargs)
    valid = exact_match(out_pallas, out_xla)
    print('Matches XLA: ', exact_match(out_pallas, out_xla))

    if not valid:
      m = jnp.zeros(out_xla[0].shape, dtype=bool)
      for ox, op in zip(out_xla, out_pallas):
        m |= ~((ox == op) | (jnp.isnan(ox) & jnp.isnan(op)))
      for ox, op in zip(out_xla, out_pallas):
        print(f'xla {ox[m]}\npallas {op[m]}')
  else:
    # Check output is valid permutation with correct relative order
    out_pallas_stable_sorted = _equiv_xla_based_sort(
        out_pallas,
        num_keys=num_keys,
        is_stable=True,
        descending=descending,
    )
    valid = exact_match(out_pallas, out_pallas_stable_sorted)
    print('out_pallas==stablesort(out_pallas): ', valid)

    narrs = len(out_pallas)
    operands_fully_sorted = _equiv_xla_based_sort(
        operand, **{**kwargs, 'num_keys': narrs}
    )
    out_pallas_fully_sorted = _equiv_xla_based_sort(
        out_pallas, **{**kwargs, 'num_keys': narrs, 'return_argsort': False}
    )
    valid_permute = exact_match(operands_fully_sorted, out_pallas_fully_sorted)
    print('out_pallas is permute of input: ', valid_permute)
    valid &= valid_permute

  if print_outputs:
    o_pallas, o_xla = _equiv_xla_based_sort(operand, **kwargs)
    print(f'Pallas: {o_pallas}\nXLA: {o_xla}')

def tests():
  ntoken = 8
  for num_operands in range(1,2):
    for num_keys in range(1, num_operands+1):
      for n in (
          2**9,
          2**8+1,
          313,
          57,
      ):
        for dtype in (
            jnp.float32,
            jnp.bfloat16,
            jnp.int32,
        ):
          operands = list(jax.random.randint(jax.random.key(0), (num_operands, ntoken,n), jnp.iinfo(jnp.int32).min, jnp.iinfo(jnp.int32).max, jnp.int32).view(dtype)[...,:n])
          for kwargs in (
              dict(),
              dict(descending=True),
              dict(return_argsort=True),
              dict(is_stable=True),
              dict(return_argsort=True, is_stable=True),
              dict(return_argsort=True, descending=True),
              dict(is_stable=True, descending=True),
              dict(return_argsort=True, is_stable=True, descending=True),
          )[:4]:
            x = operands[0]
            print(f'\n{(x.shape, x.dtype)}\n{num_operands=} {num_keys=} {kwargs=}')
            check_lax_sort_pallas(operands, num_keys=num_keys, **kwargs,
            )

if __name__ == "__main__":
  tests()
