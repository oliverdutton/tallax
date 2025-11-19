
import functools
import gzip
import json
import os
from glob import glob
import pandas as pd

import jax
import jax.numpy as jnp

from top_k import topk_pallas


k = 64
num_queries = 32
vocab_size = 201088
hidden_dim = 2880

logit_key, key_act, key_weight = jax.random.split(jax.random.key(0), 3)
x = jax.random.normal(key_act, (num_queries, hidden_dim), dtype=jnp.bfloat16)
w = jax.random.normal(key_weight, (hidden_dim, vocab_size), dtype=jnp.bfloat16)
logits = jax.random.normal(
    key_weight, (num_queries, vocab_size), dtype=jnp.float32
).astype(jnp.bfloat16)

topk_xla = jax.jit(jax.lax.top_k, static_argnames=("k",))
approx_topk_xla = jax.jit(jax.lax.approx_max_k, static_argnames=("k",))
sort_xla = jax.jit(jnp.sort)
argsort_xla = jax.jit(jnp.argsort)
@jax.jit
def add_one(x):
  return x+1


@jax.jit
@functools.partial(jax.vmap, in_axes=(0, None))
def matmul_and_topk_xla(x, w, k=k):
  logits = x @ w
  return jax.lax.top_k(logits, k)

def benchmark(_run):
  def run():
    return jax.block_until_ready(_run())
  run()
  with jax.profiler.trace("/content/"):
    run()

  path = sorted(glob("/content/plugins/profile/*/**.json.gz"), key=os.path.getmtime)[-1]
  trace = json.load(gzip.open(path))
  df = pd.DataFrame(trace["traceEvents"])
  df = df[~df.name.isna()]
  print(df[df.name.str.contains("jit_")][['name', 'dur']])

check = True

def _run():
  return (
    add_one(logits),
    topk_xla(logits, k=k),
    topk_pallas(logits, k=k, block_size=8),
    # Not exact. Runtime varies with recall, here run with default 0.95
    approx_topk_xla(logits, k=k),
  )

if check:
  benchmark(_run)
  print('topk', logits.shape, logits.dtype, k)
  print("XLA: ", topk_xla(logits, k=k))
  print("\nPallas:", topk_pallas(logits, k=k))
  print(
  [
  (topk_xla(logits, k=k)[i] == topk_pallas(logits, k=k)[i]).mean() for i in range(2)
  ]
  )
