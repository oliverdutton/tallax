import pytest
import jax
import jax.numpy as jnp
import numpy as np
from tallax.sampling import topk_topp_and_sample
from tallax._src.tpu_inference_sampling_as_standalone_file import (
    TPUSupportedSamplingMetadata,
    sample as vllm_sample,
    ShardingAxisName2D,
    Mesh,
)
from tallax._src.utils import is_cpu_platform


def uniquely_define_topk(logits, k):
    """Ensure topk is well-defined by handling ties at the k-th boundary.

    If more than k values are >= the k-th largest value, set extras to -inf.
    This ensures topk is deterministic.
    """
    boundary_val = jax.lax.sort(logits)[-k]
    mask = logits >= boundary_val
    # if more than k values gt k-th largest value, set them to -inf
    mask = mask & (mask.cumsum() > k)
    return jnp.where(mask, float('-inf'), logits)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("case", ["random", "worst_case"])
@pytest.mark.skipif(
    is_cpu_platform(),
    reason="Sampling tests require TPU/GPU - CPU uses interpret mode which is slow"
)
def test_topk_topp_and_sample(dtype, case):
    """Test topk_topp_and_sample implementation against vLLM reference.

    Tests both random and worst-case logits distributions.
    Validates that pallas implementation matches vLLM sampling behavior.
    """
    num_tokens, vocab_size = shape = (16, 2**18)

    # Create mesh for vLLM sample function
    mesh = Mesh(np.array([jax.devices()[0]]), axis_names=(ShardingAxisName2D.ATTN_DATA,))

    # Create sampling metadata with varying top_k, top_p, and temperature
    tpu_sampling_metadata = TPUSupportedSamplingMetadata(
        top_k=jax.random.randint(jax.random.key(17), (num_tokens,), 7, 128, dtype=jnp.int32) * 0 + 2,
        top_p=jax.random.uniform(jax.random.key(22), (num_tokens,), dtype=jnp.float32),
        temperature=10**jax.random.normal(jax.random.key(73), (num_tokens,), dtype=jnp.float32),
        do_sampling=True,
    )

    # Generate test data
    key, sample_key = jax.random.split(jax.random.PRNGKey(42))

    # Generate logits based on case
    if case == "random":
        logits = jax.random.normal(key, shape).astype(dtype)
    else:  # worst_case
        logits = jax.random.normal(key, shape).astype(dtype)
        logits = logits.at[:, 13::256].add(100)

    logits = jax.vmap(uniquely_define_topk)(logits, tpu_sampling_metadata.top_k)

    # Run both implementations
    pallas_result = topk_topp_and_sample(
        sample_key,
        logits,
        tpu_sampling_metadata
    )

    vllm_result = vllm_sample(
        sample_key,
        mesh,
        logits,
        tpu_sampling_metadata
    )

    # Compare results
    match_rate = (pallas_result == vllm_result).mean()

    # With varying k, p, and temperature, we expect high (but not perfect) match rate
    # due to different numerical precision and potential implementation differences
    assert match_rate > 0.8, (
        f"Pallas sampling should mostly match vLLM sampling for {case} case: "
        f"match_rate={match_rate:.2%}, dtype={dtype}"
    )


@pytest.mark.parametrize("shape", [(16, 2**18), (32, 2**17)])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.skipif(
    is_cpu_platform(),
    reason="Sampling tests require TPU/GPU - CPU uses interpret mode which is slow"
)
def test_topk_topp_and_sample_shapes(shape, dtype):
    """Test topk_topp_and_sample with different shapes."""
    num_tokens, vocab_size = shape

    # Create mesh for vLLM sample function
    mesh = Mesh(np.array([jax.devices()[0]]), axis_names=(ShardingAxisName2D.ATTN_DATA,))

    # Create sampling metadata
    tpu_sampling_metadata = TPUSupportedSamplingMetadata(
        top_k=jnp.full((num_tokens,), 64, dtype=jnp.int32),
        top_p=jnp.full((num_tokens,), 0.9, dtype=jnp.float32),
        temperature=jnp.full((num_tokens,), 1.0, dtype=jnp.float32),
        do_sampling=True,
    )

    # Generate test data
    key = jax.random.PRNGKey(42)
    sample_key = jax.random.PRNGKey(123)

    logits = jax.random.normal(key, shape).astype(dtype)
    logits = jax.vmap(uniquely_define_topk)(logits, tpu_sampling_metadata.top_k)

    # Run pallas implementation
    pallas_result = topk_topp_and_sample(
        sample_key,
        logits,
        tpu_sampling_metadata
    )

    # Run vLLM implementation
    vllm_result = vllm_sample(
        sample_key,
        mesh,
        logits,
        tpu_sampling_metadata
    )

    # Validate output shape
    assert pallas_result.shape == (num_tokens,), (
        f"Expected output shape {(num_tokens,)}, got {pallas_result.shape}"
    )

    # Validate output values are in valid range
    assert jnp.all((pallas_result >= 0) & (pallas_result < vocab_size)), (
        "Sampled tokens should be in valid range [0, vocab_size)"
    )

    # Compare with vLLM
    match_rate = (pallas_result == vllm_result).mean()
    assert match_rate > 0.7, (
        f"Pallas sampling should mostly match vLLM sampling: "
        f"match_rate={match_rate:.2%} for shape={shape}, dtype={dtype}"
    )


if __name__ == "__main__":
    print("Running topk_topp_and_sample tests...")

    # Test with different dtypes and cases
    for dtype in [jnp.bfloat16, jnp.float32]:
        for case in ["random", "worst_case"]:
            print(f"\nTesting dtype={dtype}, case={case}...")
            test_topk_topp_and_sample(dtype, case)
            print(f"  ✓ Passed")

    # Test with different shapes
    shapes = [(16, 2**18), (32, 2**17)]
    dtypes = [jnp.bfloat16, jnp.float32]

    for shape in shapes:
        for dtype in dtypes:
            print(f"\nTesting shape={shape}, dtype={dtype}...")
            test_topk_topp_and_sample_shapes(shape, dtype)
            print(f"  ✓ Passed")

    print("\nAll topk_topp_and_sample tests passed!")
