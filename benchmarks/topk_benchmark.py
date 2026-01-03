# @title Speculative Decoding Top-k
"""
!rm -rf tallax
branch = 'main'
!git clone -q -b {branch} --single-branch https://github.com/oliverdutton/tallax.git && cd tallax && pip install -q .[tpu]
"""

import jax
import jax.numpy as jnp
from tallax import tax
from tallax.tax.test_utils import benchmark, verify_topk_output

topk_xla = jax.jit(jax.lax.top_k, static_argnames=("k",))


def benchmark_topk(shape, k, dtype, case, seed):
  """Test topk_topp_and_sample implementation against vLLM reference.

  Tests both random and worst-case logits distributions.
  Validates that pallas implementation matches vLLM sampling behavior exactly.
  """
  num_tokens, vocab_size = shape

  # Split main seed into all needed keys
  key = jax.random.PRNGKey(seed)

  # Generate logits based on case
  logits = jax.random.normal(key, shape).astype(dtype)
  if case == "worst_case":
    logits = logits.at[:, 13::256].add(100)

  # Run both implementations
  def _run():
    return (
      tax.top_k(logits, k=k),
      topk_xla(logits, k=k),
    )

  pallas_result, _ = _run()
  benchmark(_run)

  valid = verify_topk_output(logits, pallas_result, axis=1)
  assert valid.all(), (
    f"Top-k validation failed for shape {shape}, dtype {dtype}, k {k}"
  )


if __name__ == "__main__":
  print("Running topk_topp_and_sample tests...")

  shapes = [(16, 2**15), (128, 2**15)]
  k = 5
  dtypes = [
    jnp.bfloat16,
  ]
  cases = ["random", "worst_case"]
  seeds = [
    42,
  ]

  for shape in shapes:
    for dtype in dtypes:
      for case in cases:
        for seed in seeds:
          print(
            f"\nTesting shape={shape}, dtype={dtype}, case={case}, seed={seed}..."
          )
          benchmark_topk(shape, k, dtype, case, seed)

  print("\nAll topk checks passed!")
