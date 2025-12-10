import functools
import gzip
import json
import os
from glob import glob
import jax
import jax.numpy as jnp
import pandas as pd
import pytest

from tallax import tax
from tallax.utils import is_cpu_platform


@jax.jit
def exact_match(xs, ys):
  """Check if two pytrees match exactly (including NaN positions)."""
  return jnp.array(jax.tree.leaves(
      jax.tree.map(lambda x, y: jnp.array_equal(x, y, equal_nan=True), xs, ys)
  )).all()


def verify_sort_output(
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
    out_xla = tax.sort_xla_equivalent(operand, **kwargs_for_xla)
    valid = bool(exact_match(out_pallas, out_xla))

    if not valid:
      m = jnp.zeros(out_xla[0].shape, dtype=bool)
      for ox, op in zip(out_xla, out_pallas):
        m |= ~((ox == op) | (jnp.isnan(ox) & jnp.isnan(op)))
      debug_msg = []
      for ox, op in zip(out_xla, out_pallas):
        debug_msg.append(f'xla {ox[m]}\npallas {op[m]}')
      debug_output = '\n'.join(debug_msg)
      error_msg = f"Pallas output does not match XLA output for stable sort:\n{debug_output}"
    else:
      error_msg = "Pallas output does not match XLA output for stable sort"

    assert valid, error_msg

  else:
    # Check output is valid permutation with correct relative order
    out_pallas_stable_sorted = tax.sort_xla_equivalent(
        out_pallas,
        num_keys=num_keys,
        is_stable=True,
        descending=descending,
        interpret=interpret,
    )
    valid = bool(exact_match(out_pallas, out_pallas_stable_sorted))
    if not valid:
      m = jnp.zeros(out_pallas_stable_sorted[0].shape, dtype=bool)
      for ox, op in zip(out_pallas_stable_sorted, out_pallas):
        m |= ~((ox == op) | (jnp.isnan(ox) & jnp.isnan(op)))
      debug_msg = []
      for ox, op in zip(out_pallas_stable_sorted, out_pallas):
        debug_msg.append(f'sorted {ox[m]}\npallas {op[m]}')
      debug_output = '\n'.join(debug_msg)
      error_msg = f"Pallas output is not sorted:\n{debug_output}"
    else:
      error_msg = "out_pallas must be sorted (verified by re-sorting stably)"

    assert valid, error_msg

    narrs = len(out_pallas)
    kwargs_for_xla = kwargs.copy()
    operands_fully_sorted = tax.sort_xla_equivalent(
        operand, **{**kwargs_for_xla, 'num_keys': narrs}
    )
    out_pallas_fully_sorted = tax.sort_xla_equivalent(
        out_pallas, **{**kwargs_for_xla, 'num_keys': narrs, 'return_argsort': False}
    )
    valid_permute = bool(exact_match(operands_fully_sorted, out_pallas_fully_sorted))
    assert valid_permute, "out_pallas is not a valid permutation of input"
    valid &= valid_permute

  if print_outputs:
    o_pallas, o_xla = tax.sort_xla_equivalent(operand, **kwargs)
    print(f'Pallas: {o_pallas}\nXLA: {o_xla}')


def verify_topk_output(x, outs, axis=1):
    """Validate top-k outputs for correctness.

    Args:
        x: Input array (must be 2D)
        outs: Tuple of (values, indices) from top-k
        axis: Axis along which top-k was computed (0 or 1, default 1)

    Returns:
        Boolean array indicating if the top-k output is valid for each batch element

    Raises:
        ValueError: If x is not 2D
    """
    if x.ndim != 2:
        raise ValueError(f"verify_topk_output only supports 2D inputs, got {x.ndim}D")

    # If axis == 0, transpose so we always work with axis=1
    if axis == 0:
        x = x.T
        out_vals, out_indexs = outs
        out_vals = out_vals.T
        out_indexs = out_indexs.T
    else:
        out_vals, out_indexs = outs

    # Now verify along axis=1 for each row
    @jax.vmap
    def verify_row(x_row, vals_row, idxs_row):
        """Verify a single row."""
        x_sorted = jnp.sort(x_row, descending=True)

        k = len(vals_row)
        n = len(x_row)
        valid = True

        # actual values must match
        valid &= (vals_row == x_sorted[:k]).all()

        # indices map to values correctly
        valid &= (x_row[idxs_row] == vals_row).all()

        # indices are all in bounds and unique
        i = jnp.unique(idxs_row, size=k, fill_value=-1)
        valid &= ((i >= 0) & (i < n)).all()
        return valid

    return verify_row(x, out_vals, out_indexs)


def benchmark(_run):
  """Benchmark function and print timing from profiler trace."""
  def run():
    return jax.block_until_ready(_run())

  # Warmup
  run()

  tmpdir = "."
  with jax.profiler.trace(tmpdir):
    run()

  # Find trace file
  files = glob(f"{tmpdir}/plugins/profile/*/**.json.gz", recursive=True)
  if not files:
    print("No trace file generated.")
    return

  path = sorted(files, key=os.path.getmtime)[-1]
  try:
    with gzip.open(path, 'rb') as f:
      trace = json.load(f)
  except Exception as e:
    print(f"Failed to load trace: {e}")
    return

  if "traceEvents" not in trace:
    print("No traceEvents in trace.")
    return

  df = pd.DataFrame(trace["traceEvents"])
  if df.empty or 'name' not in df.columns:
    print("Trace dataframe empty or no name column.")
    return

  df = df[~df.name.isna()]
  df['name'] = df.name.apply(lambda s: s.split('(')[0])

  # Look for JIT compiled functions
  mask = df.name.str.contains("jit_")
  res = df[mask][['name', 'dur']]

  if not res.empty:
    print(res.to_string(index=False))
  else:
    print("No jit functions found in trace.")
