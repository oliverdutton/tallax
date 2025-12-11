"""
Fixed TPU sampling script with corrected imports and API usage.

Main fixes:
1. Changed import from tallax.tax.fused_sampling to tallax._src.sampling
2. Changed import from tallax.test_utils to tallax._src.test_utils
3. Removed mesh parameter from pallas_sample function (not needed in new API)
4. Use sample function from tallax._src.sampling directly
5. Fixed log2 import path
"""

# Note: These clone and install commands would be run in a Colab notebook cell
# !rm -rf tallax
# branch = 'main'
# !git clone -q -b {branch} --single-branch https://github.com/oliverdutton/tallax.git && cd tallax && pip install -q .[tpu]
# !git clone -b sample_standalone --single-branch https://github.com/oliverdutton/tpu-inference.git

import sys
import os
# Add the repo path to your system path
sys.path.append(os.path.abspath("/content/tpu-inference"))
from sample_standalone import TPUSupportedSamplingMetadata, sample as vllm_sample, ShardingAxisName2D, Mesh, topp_mask

# Fixed imports from tallax
from tallax.tax import top_dynamic_k
from tallax._src.bitonic_topk import bitonic_topk
from tallax._src.sampling import top_p_and_sample, sample  # Fixed: was tallax.tax.fused_sampling
from tallax._src.utils import log2  # Fixed: explicit import from _src.utils
from tallax._src.test_utils import benchmark  # Fixed: was tallax.test_utils
import jax
from jax import numpy as jnp
import numpy as np
import functools
from jax.experimental import pallas as pl

def pallas_sample(rng_key, logits, tpu_sampling_metadata):
  """
  Fixed version: removed mesh parameter, uses internal API correctly.
  This function now matches the signature expected by the sample function in tallax.
  """
  num_bins = 256
  vocab_size = logits.shape[1]
  topk_logits, topk_idxs = top_dynamic_k(
    logits,
    k=tpu_sampling_metadata.top_k,
    max_k=128,
    guarantee_convergence=False,
    num_bins=num_bins,
    bins_topm_schedule=tuple(sorted(set(min(v, pl.cdiv(logits.shape[1], num_bins)) for v in (5,9)))),
    replace_val=-1e12)[:2]
  return top_p_and_sample(
    topk_logits, topk_idxs,
    rng_key,
    top_p=tpu_sampling_metadata.top_p,
    temperature=tpu_sampling_metadata.temperature,
    vocab_size=vocab_size,
    replace_val=-1e12)

shape = (16, 1024*4)
num_tokens, vocab_size = shape
print(shape)

tpu_sampling_metadata = TPUSupportedSamplingMetadata(
  top_k=jax.random.randint(jax.random.key(17), (num_tokens,), 7, 128, dtype=jnp.int32),
  top_p=jax.random.uniform(jax.random.key(22), (num_tokens,), dtype=jnp.float32),
  temperature=10**jax.random.normal(jax.random.key(73), (num_tokens,), dtype=jnp.float32),
  do_sampling=True,
)
mesh = Mesh(np.array([jax.devices()[0]]), axis_names=(ShardingAxisName2D.ATTN_DATA,))

# Generate test data
key, sample_key = jax.random.split(jax.random.PRNGKey(4267))
total_size = num_tokens * vocab_size

def seed_vals(key):
  idx = jax.random.randint(key, (128,), 0, vocab_size)
  return jnp.zeros((vocab_size,), jnp.bfloat16).at[idx].set(1.0)

logits = jax.vmap(seed_vals)(jax.random.split(key, num_tokens))
logits = jax.random.normal(key, shape).astype(logits.dtype)
logits_worst_case = logits.at[:,::512].add(15)
assert logits.shape == shape

def make_topk_unique(logits, k):
  boundary_val = jax.lax.sort(logits)[-k]
  mask = logits >= boundary_val
  # if more than k values gt k-th largest value, set them to -inf. this way topk is well defined
  mask = mask & (mask.cumsum() > k)
  return jnp.where(mask, float('-inf'), logits)

logits = jax.vmap(make_topk_unique)(logits, tpu_sampling_metadata.top_k)
idxs = jax.lax.broadcasted_iota(jnp.int32, logits.shape, 1)

def _run():
  return (*tuple(
    # Fixed: removed mesh parameter from pallas_sample calls
    f(sample_key, v, tpu_sampling_metadata)
    for v in (logits, logits_worst_case)
    for f in (
         pallas_sample,
         lambda rng, logits, meta: vllm_sample(rng, mesh, logits, meta),  # vllm_sample still needs mesh
    )
  ),
    bitonic_topk([logits, idxs], k=128),
  )

benchmark(_run)
for v in _run():
  print(v)
a, b, *_ = _run()
print((a==b).mean())
print(tpu_sampling_metadata.temperature)

# Alternative: Use the all-in-one sample function from tallax
# This is the recommended approach - it handles top-k and sampling internally
def _run_with_tallax_sample():
  """Example using the unified sample function from tallax._src.sampling"""
  return sample(sample_key, logits, tpu_sampling_metadata)

print("\n--- Using tallax.sample (recommended) ---")
tallax_result = _run_with_tallax_sample()
print(tallax_result)
