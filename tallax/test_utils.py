import gzip
import json
import os
from glob import glob
import jax
import jax.numpy as jnp
import pandas as pd


@jax.vmap
def check_topk_out(x, outs):
    """Validate top-k outputs for correctness.

    Args:
        x: Input array (1D per vmap)
        outs: Tuple of (values, indices) from top-k

    Returns:
        Boolean indicating if the top-k output is valid
    """
    assert x.ndim == 1
    out_vals, out_indexs = outs
    x_sorted = jnp.sort(x, descending=True)

    k = len(out_vals)
    n = len(x)
    valid = True

    # actual values must match
    valid &= (out_vals == x_sorted[:k]).all()

    # indices map to values correctly
    valid &= (x[out_indexs] == out_vals).all()

    # indices are all in bounds and unique
    i = jnp.unique(out_indexs, size=k, fill_value=-1)
    valid &= ((i >= 0) & (i < n)).all()
    return valid


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
