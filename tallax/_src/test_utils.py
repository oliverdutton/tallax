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
from tallax._src.utils import is_cpu_platform
from tallax._src.sort import xla_equivalent_sort


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
    out_xla = xla_equivalent_sort(operand, **kwargs_for_xla)
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
    out_pallas_stable_sorted = xla_equivalent_sort(
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
    operands_fully_sorted = xla_equivalent_sort(
        operand, **{**kwargs_for_xla, 'num_keys': narrs}
    )
    out_pallas_fully_sorted = xla_equivalent_sort(
        out_pallas, **{**kwargs_for_xla, 'num_keys': narrs, 'return_argsort': False}
    )
    valid_permute = bool(exact_match(operands_fully_sorted, out_pallas_fully_sorted))
    assert valid_permute, "out_pallas is not a valid permutation of input"
    valid &= valid_permute

  if print_outputs:
    o_pallas, o_xla = xla_equivalent_sort(operand, **kwargs)
    print(f'Pallas: {o_pallas}\nXLA: {o_xla}')


def verify_topk_output(x, outs, axis=1, approximate=False):
    """Validate top-k outputs for correctness.

    Args:
        x: Input array (must be 2D)
        outs: Tuple of (values, indices) from top-k (both must be 2D)
        axis: Axis along which top-k was computed (0 or 1, default 1)
        approximate: If True, return float % of vals >= threshold (0.0 if indices invalid).
                     If False, return boolean exact match (default False)

    Returns:
        If approximate=False: Boolean array indicating validity for each batch element
        If approximate=True: Float array with % of values of top-k present (0.0 if indices fail)
    """
    if x.ndim != 2:
        raise ValueError(f"verify_topk_output only supports 2D inputs, got {x.ndim}D")

    out_vals, out_indexs = outs

    if out_vals.ndim != 2 or out_indexs.ndim != 2:
        raise ValueError(f"verify_topk_output requires 2D outputs, got values.ndim={out_vals.ndim}, indices.ndim={out_indexs.ndim}")

    batch_axis = 1 - axis

    @functools.partial(jax.vmap, in_axes=batch_axis)
    def verify_slice(x_slice, vals_slice, idxs_slice):
        k = len(vals_slice)
        n = len(x_slice)

        true_topk_vals = jax.lax.top_k(x_slice, k)[0]

        indices_mapping_valid = (x_slice[idxs_slice] == vals_slice).all()
        i = jnp.unique(idxs_slice, size=k, fill_value=-1)
        indices_bounds_valid = ((i >= 0) & (i < n)).all()
        indices_valid = indices_mapping_valid & indices_bounds_valid

        if approximate:
            threshold = true_topk_vals[-1]
            # due to ties at the topk boundary we have to be careful here
            vals_recall = (
            # how many values definitely in topk, with a max topk inclusion number at the threshold
            (vals_slice > threshold).sum() + jnp.minimum(
            (true_topk_vals == threshold).sum(), (vals_slice == threshold).sum())
            ) / k
            return jnp.where(indices_valid, vals_recall, 0.0)
        else:
            vals_valid = (vals_slice == true_topk_vals).all()
            return vals_valid & indices_valid

    return verify_slice(x, out_vals, out_indexs)


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
  mask = df.name.str.startswith("jit_")
  res = df[mask][['name', 'dur']]

  if not res.empty:
    print(res.to_string(index=False))
  else:
    print("No jit functions found in trace.")
