import gzip
import json
import os
from glob import glob
import jax
import pandas as pd

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
  res = df[['name', 'dur']]

  if not res.empty:
    print(res.to_string(index=False))
  else:
    print("No jit functions found in trace.")
