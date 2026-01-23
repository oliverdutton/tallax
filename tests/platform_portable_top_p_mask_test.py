import pytest
import jax
import jax.numpy as jnp
import numpy as np
from tallax.tax.platform_portable_top_p import platform_portable_top_p
from tallax.vllm.tpu_inference_sampling_as_standalone_file import (
  topp_mask as tpu_inference_top_p_mask,
)


@pytest.mark.parametrize(
  "shape",
  [
    (8, 128),
    (16, 256),
    (13, 167),
    (21, 128),
    (256, 128),
    (137, 17),
    (137, 193),
  ],
)
@pytest.mark.parametrize("seed", [42, 123, 456])
@pytest.mark.parametrize("p_threshold", [0.001, 0.1, 0.5, 0.999, 1.0, None])
def test_top_p_mask(shape, seed, p_threshold):
  """Test platform_portable_top_p for exact match against tpu_inference_top_p_mask.
  
  Strategy:
  1. Generate random logits (f32)
  2. Apply platform_portable_top_p (which handles sorting/unsorting internally if needed, or works on unsorted)
     Update: The implementation works on unsorted logits directly.
  3. Apply tpu_inference_top_p_mask to unsorted logits (per-batch element)
  4. Compare results (should match exactly in f32)
  """
  key = jax.random.key(seed)
  key, logits_key, p_key = jax.random.split(key, 3)

  # Generate random logits (f32)
  logits = jax.random.normal(logits_key, shape, dtype=jnp.float32)

  # Generate p values: None means random uniform, otherwise use fixed threshold
  if p_threshold is None:
    p_array = jax.random.uniform(p_key, shape[:1], dtype=jnp.float32)
  else:
    p_array = jnp.full(shape[:1], p_threshold, dtype=jnp.float32)

  replace_val = -1e12

  # Apply platform_portable_top_p
  result_portable = platform_portable_top_p(
    logits=logits,
    top_p=p_array,
    replace_val=replace_val,
  )

  # Apply tpu_inference_top_p_mask per batch element (expects unsorted logits and scalar p)
  result_tpu_inference = tpu_inference_top_p_mask(
      logits, p_array, replace_val
  )
  # Compare results (should match exactly in f32)
  np.testing.assert_array_equal(
    result_portable,
    result_tpu_inference,
    err_msg=f"platform_portable_top_p should match tpu_inference_top_p_mask for shape={shape}, seed={seed}, p={p_threshold}",
  )


if __name__ == "__main__":
  print("Running top_p_mask tests...")
  shapes = [(8, 128), (16, 256), (13, 167), (32, 128)]
  seeds = [42, 123, 456]
  p_thresholds = [0.001, 0.1, 0.5, 0.999, 1.0, None]

  for shape in shapes:
    for seed in seeds:
      for p_threshold in p_thresholds:
        print(
          f"Testing shape={shape}, seed={seed}, p_threshold={p_threshold}..."
        )
        test_top_p_mask(shape, seed, p_threshold)
        print("  ✓ Passed")

  print("\nAll top_p_mask tests passed!")
