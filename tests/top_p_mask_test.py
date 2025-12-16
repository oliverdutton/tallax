import pytest
import jax
import jax.numpy as jnp
import numpy as np
from tallax._src.top_p_and_sample import top_p_mask
from tallax._src.tpu_inference_sampling_as_standalone_file import topp_mask


@pytest.mark.parametrize("shape", [(8, 128), (16, 256), (13, 167), (256, 256), (173, 195)])
@pytest.mark.parametrize("p", [0.001, 0.1, 0.5, 0.999, 1.0])
def test_top_p_mask(shape, p):
    """Test top_p_mask for exact match against topp_mask from tpu_inference_sampling."""
    key = jax.random.key(42)

    # Generate random logits (f32)
    logits = jax.random.normal(key, shape, dtype=jnp.float32)

    # Sort logits in descending order for top_p_mask (axis=1)
    sorted_logits_descending = jnp.sort(logits, axis=1, descending=True)

    # Transpose for top_p_mask (expects axis=0)
    sorted_logits_transposed = sorted_logits_descending.T

    replace_val = -1e12

    # Apply top_p_mask (axis=0 on transposed logits)
    p_array = jnp.full((shape[0],), p, dtype=jnp.float32)
    result_top_p_mask = top_p_mask(
        topk_logits=sorted_logits_transposed,
        p=p_array,
        replace_val=replace_val,
        axis=0
    )

    # Transpose back to (batch, vocab) for comparison
    result_top_p_mask = result_top_p_mask.T

    # Apply topp_mask (expects unsorted logits)
    result_topp_mask = topp_mask(logits, p, replace_val)

    # Sort both results for comparison
    result_top_p_mask_sorted = jnp.sort(result_top_p_mask, axis=1, descending=True)
    result_topp_mask_sorted = jnp.sort(result_topp_mask, axis=1, descending=True)

    # Should match exactly (even in f32)
    np.testing.assert_array_equal(result_top_p_mask_sorted, result_topp_mask_sorted,
        err_msg=f"top_p_mask should match topp_mask for shape={shape}, p={p}")


if __name__ == "__main__":
    print("Running top_p_mask tests...")
    test_top_p_mask((8, 128), 0.5)
    test_top_p_mask((16, 256), 0.9)
    print("top_p_mask tests passed!")
