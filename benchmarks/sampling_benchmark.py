

#@title vLLM Sampling
'''
!rm -rf tallax
branch = 'main'
!git clone -q -b {branch} --single-branch https://github.com/oliverdutton/tallax.git && cd tallax && pip install -q .[tpu]
'''

import jax
from jax import numpy as jnp
import numpy as np
import functools
from jax.experimental import pallas as pl

from tallax._src.sampling import sample as _pallas_sample
from tallax._src.test_utils import benchmark

from tallax._src.tpu_inference_sampling_as_standalone_file import TPUSupportedSamplingMetadata, sample as vllm_sample, ShardingAxisName2D, Mesh, topp_mask

def uniquely_define_topk(logits, k):
  boundary_val = jax.lax.sort(logits)[-k]
  mask = logits >= boundary_val
  #jax.debug.print('{} elements >= k boundary', mask.sum())
  # if more than k values gt k-th largest value, set them to -inf. this way topk is well defined
  mask = mask & (mask.cumsum() > k)
  return jnp.where(mask, float('-inf'), logits)

@functools.partial(
    jax.jit,
    static_argnames=["mesh"],
)
def pallas_sample(rng_key, mesh, logits, tpu_sampling_metadata):
  '''Wrapper to match tpu-inference sample function'''
  del mesh
  return _pallas_sample(rng_key, logits, tpu_sampling_metadata)

mesh = Mesh(np.array([jax.devices()[0]]), axis_names=(ShardingAxisName2D.ATTN_DATA,))

num_tokens, vocab_size = shape = (16, 2**18)
print(shape)

tpu_sampling_metadata = TPUSupportedSamplingMetadata(
  top_k=jax.random.randint(jax.random.key(17), (num_tokens,), 7, 128, dtype=jnp.int32),
  top_p=jax.random.uniform(jax.random.key(22), (num_tokens,), dtype=jnp.float32),
  temperature=10**jax.random.normal(jax.random.key(73), (num_tokens,), dtype=jnp.float32),
  do_sampling=True,
)

# Generate test data
key, sample_key = jax.random.split(jax.random.PRNGKey(4267))
total_size = num_tokens * vocab_size

logits = jax.random.normal(key, shape).astype(jnp.bfloat16)
logits_worst_case = logits.at[:,::256].add(100)
logits_top_4 = logits.at[:,:128].add(100)
logits_top_4_8 = logits.at[:8,:128].add(100).at[8:,:256*8:16].add(100)
logits_top_8 = logits.at[:,:256*8:16].add(100)

logits_cases = (logits, logits_top_4, logits_top_4_8, logits_top_8, logits_worst_case)

logits_cases = tuple(jax.vmap(uniquely_define_topk)(x, tpu_sampling_metadata.top_k) for x in logits_cases)
#idxs = jax.lax.broadcasted_iota(jnp.int32, logits.shape, 1)

def _run():
  return (*tuple(
    f(sample_key, mesh, v, tpu_sampling_metadata)
    for v in logits_cases
    for f in (
         pallas_sample,
         vllm_sample,
    )
  ),
    #bitonic_topk([logits, idxs], k=NUM_LANES),
  )

benchmark(_run)

# Seeing sampled tokens is believing
for v in _run():
  print(v)
outs = _run()
# Check sampled token matches
# We use varying k and temperatures of 10**rand so that sometimes random gumbel noise dominates, sometimes logits values dominates
# Similarly, varying p threshold in top-p
print(['{:.2f}'.format((a==b).mean()) for a, b in zip(outs[::2], outs[1::2])])
#print(tpu_sampling_metadata.temperature)