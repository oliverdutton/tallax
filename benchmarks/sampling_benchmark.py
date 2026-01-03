# @title vLLM Sampling
"""
!rm -rf tallax
branch = 'main'
!git clone -q -b {branch} --single-branch https://github.com/oliverdutton/tallax.git && cd tallax && pip install -q .[tpu]
"""

import jax
import jax.numpy as jnp
import numpy as np
from tallax.vllm import topk_topp_and_sample
from tallax.vllm.tpu_inference_sampling_as_standalone_file import (
  TPUSupportedSamplingMetadata,
  sample as vllm_sample,
  ShardingAxisName2D,
  Mesh,
)
from tallax.tax.test_utils import benchmark, uniquely_define_topk


def benchmark_topk_topp_and_sample(shape, k, dtype, case, seed):
  """Test topk_topp_and_sample implementation against vLLM reference.

  Tests both random and worst-case logits distributions.
  Validates that pallas implementation matches vLLM sampling behavior exactly.
  """
  num_tokens, vocab_size = shape

  # Create mesh for vLLM sample function
  mesh = Mesh(
    np.array([jax.devices()[0]]), axis_names=(ShardingAxisName2D.ATTN_DATA,)
  )

  # Split main seed into all needed keys
  key = jax.random.PRNGKey(seed)
  key, topk_key, topp_key, temp_key, logits_key, sample_key = jax.random.split(
    key, 6
  )

  # Create sampling metadata with varying top_k, top_p, and temperature
  # We use varying temperatures of 10**normal(0,1) so that sometimes random gumbel noise dominates,
  # sometimes logits values dominates. Similarly, varying p threshold in top-p
  tpu_sampling_metadata = TPUSupportedSamplingMetadata(
    # all k
    top_k=k * jnp.ones((num_tokens,), dtype=jnp.int32),
    top_p=jax.random.uniform(topp_key, (num_tokens,), dtype=jnp.float32),
    temperature=10
    ** jax.random.normal(temp_key, (num_tokens,), dtype=jnp.float32),
    do_sampling=True,
  )

  # Generate logits based on case
  logits = jax.random.normal(logits_key, shape).astype(dtype)
  if case == "worst_case":
    logits = logits.at[:, 13::256].add(100)

  logits = jax.vmap(uniquely_define_topk)(logits, tpu_sampling_metadata.top_k)

  # Run both implementations
  def _run():
    pallas_results = [
      topk_topp_and_sample(sample_key, logits, tpu_sampling_metadata, max_k=k),
    ]

    vllm_result = vllm_sample(sample_key, mesh, logits, tpu_sampling_metadata)
    return (pallas_results, vllm_result)

  pallas_results, vllm_result = _run()
  benchmark(_run)

  # Compare results - expect exact match
  for pallas_result in pallas_results:
    np.testing.assert_array_equal(
      pallas_result,
      vllm_result,
      err_msg=f"Pallas sampling should exactly match vLLM sampling for "
      f"shape={shape}, dtype={dtype}, case={case}, seed={seed}",
    )


if __name__ == "__main__":
  print("Running topk_topp_and_sample tests...")

  shapes = [(16, 2**18), (128, 2**18)]
  k = 64
  dtypes = [
    jnp.bfloat16,
  ]
  cases = ["random", "worst_case"]
  seeds = [
    42,
    123,
    456,
  ]

  for shape in shapes:
    for dtype in dtypes:
      for case in cases:
        for seed in seeds:
          print(
            f"\nTesting shape={shape}, dtype={dtype}, case={case}, seed={seed}..."
          )
          benchmark_topk_topp_and_sample(shape, k, dtype, case, seed)
          print("  ✓ Passed")

  print("\nAll topk_topp_and_sample tests passed!")
