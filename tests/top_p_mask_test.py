import jax
import jax.numpy as jnp
import numpy as np
from tallax._src.top_p_and_sample import top_p_mask
from tallax._src.tpu_inference_sampling_as_standalone_file import topp_mask


def test_top_p_mask(shape, p):
    """Test top_p_mask for exact match against topp_mask from tpu_inference_sampling.

    Strategy:
    1. Sort input and get argsort indices (axis=1)
    2. Apply top_p_mask to sorted input
    3. Reverse argsort to return to original order
    4. Apply topp_mask to unsorted input
    5. Compare results (should match exactly in f32)
    """
    key = jax.random.key(42)

    # Generate random logits (f32)
    logits = jax.random.normal(key, shape, dtype=jnp.float32)

    replace_val = -1e12

    # Sort logits in descending order and get indices (axis=1)
    sort_indices = jnp.argsort(logits, axis=1, descending=True)
    sorted_logits = jnp.take_along_axis(logits, sort_indices, axis=1)

    # Transpose for top_p_mask (expects axis=0)
    sorted_logits_transposed = sorted_logits.T

    # Apply top_p_mask (axis=0 on transposed logits)
    p_array = jnp.full((shape[0],), p, dtype=jnp.float32)
    result_top_p_mask_sorted = top_p_mask(
        topk_logits=sorted_logits_transposed,
        p=p_array,
        replace_val=replace_val,
        axis=0
    )

    # Transpose back to (batch, vocab)
    result_top_p_mask_sorted = result_top_p_mask_sorted.T

    # Reverse the argsort to get back to original order
    # Create inverse permutation
    inverse_sort_indices = jnp.argsort(sort_indices, axis=1)
    result_top_p_mask_original_order = jnp.take_along_axis(
        result_top_p_mask_sorted, inverse_sort_indices, axis=1
    )

    # Apply topp_mask (expects unsorted logits)
    result_topp_mask = topp_mask(logits, p, replace_val)

    # Compare results in original order (should match exactly in f32)
    np.testing.assert_array_equal(result_top_p_mask_original_order, result_topp_mask,
        err_msg=f"top_p_mask should match topp_mask for shape={shape}, p={p}")


if __name__ == "__main__":
    print("Running top_p_mask tests...")
    shapes = [(8, 128), (16, 256), (13, 167), (256, 256), (173, 195)]
    ps = [0.001, 0.1, 0.5, 0.999, 1.0]

    for shape in shapes:
        for p in ps:
            print(f"Testing shape={shape}, p={p}...")
            test_top_p_mask(shape, p)
            print(f"  ✓ Passed")

    print("\nAll top_p_mask tests passed!")
