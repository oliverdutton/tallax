"""Simple smoke test for platform-portable top-p."""

import jax
jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
from tallax.tax.high_precision_uint import HighPrecisionUInt
from tallax.tax.platform_portable_top_p import platform_portable_top_p

print("Testing HighPrecisionUInt...")
x = jnp.array([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=jnp.int32)
hp = HighPrecisionUInt.from_i32_array(x)
sum_hp = hp.sum_dim1()
sums = sum_hp.to_f32().squeeze()
print(f"  Sums: {sums} (expected [10, 100])")
assert jnp.allclose(sums, jnp.array([10, 100])), "Sum failed!"

print("\nTesting platform_portable_top_p...")
logits = jnp.array([[1., 2., 3., 4., 5.]], dtype=jnp.float32)
result = platform_portable_top_p(logits, top_p=0.9)
mask = result != -1e12
num_kept = mask.sum()
print(f"  Kept {num_kept}/5 values (expected 2-4)")
assert num_kept > 0 and num_kept < 5, "Masking failed!"

print("\n✅ All tests passed!")
