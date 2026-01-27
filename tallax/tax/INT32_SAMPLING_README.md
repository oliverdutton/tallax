# Int32 Sampling for Numerically Stable Token Selection

## Overview

This module implements token sampling using int32 arithmetic instead of float32 probabilities to avoid overflow and precision issues in cumulative sum computations. This is particularly important for:

- Long sequences where float32 cumsum can lose precision
- TPU inference where int32 operations are efficient
- Scenarios where reproducible, exact sampling is required

## Problem Statement

Traditional top-p sampling uses float32 probabilities:

```python
# Traditional approach
probs = softmax(logits)           # float32 probabilities
cumsum_probs = cumsum(probs)      # Can lose precision for long sequences
threshold_idx = find_where(cumsum_probs >= p)
```

Issues with float32:
1. **Precision loss**: Cumulative sum of many small floats can lose precision
2. **Overflow risk**: Sum of exp(logits) can overflow for extreme values
3. **Non-determinism**: Float operations can have slight variations across hardware

## Solution: Int32 Sampling

Instead of working with float probabilities, we:

1. Convert logits to int32 weights
2. Use int32 cumulative sum (no precision loss)
3. Use binary search to find boundaries (latency hiding on TPU)
4. Sample uniformly from int32 range

### Algorithm

Given logits and example weights `[1, 3, 5, 3, 7, 1]`:

```
Step 1: Convert logits to int32 weights
logits = [-1.0, 0.5, 1.2, 0.5, 1.8, -1.0]
↓ exp(logits - max) scaled to int32 range
weights = [53687091, 161061274, 322122547, 161061274, 483183820, 53687091]

Step 2: Compute cumulative sum in int32
cumsum = [53687091, 214748365, 536870912, 697932186, 1181116006, 1234803097]

Step 3: Find top-p boundary using two-pass comparison
p = 0.5, total = 1234803097
threshold = 0.5 * 1234803097 = 617401548
Pass 1: Compute threshold
Pass 2: Count cumsum < threshold → boundary_idx = 2
boundary_sum = cumsum[2] = 536870912

Step 4: Sample random int32 from [0, boundary_sum)
random_int = 123456789  (uniformly sampled)

Step 5: Find token using two-pass comparison
Pass 1: threshold = random_int
Pass 2: Count cumsum <= random_int → token_idx
cumsum[0] = 53687091 <= 123456789 < cumsum[1] = 214748365
Result: token 1
```

## Example: [1, 3, 5, 3, 7, 1]

### Simplified Example (small weights for clarity)

```
Weights:   [1,   3,   5,   3,   7,   1]
Cumsum:    [1,   4,   9,  12,  19,  20]
Total sum: 20

Top-p = 0.5 (50% probability mass):
  Pass 1: Threshold = 20 * 0.5 = 10
  Pass 2: Count cumsum < 10 → [1,4,9] are < 10 → boundary_idx = 3
  But we want last included, so boundary_idx = 2
  Boundary sum = cumsum[2] = 9

Sampling:
  Random int: r = 7 (from [0, 9))

  Two-pass token selection:
    Pass 1: threshold = 7
    Pass 2: Count cumsum <= 7
      cumsum[0] = 1 <= 7?  Yes (count = 1)
      cumsum[1] = 4 <= 7?  Yes (count = 2)
      cumsum[2] = 9 <= 7?  No  (count = 2)
    First exceeding index = 2

  Selected token: 2
```

### Token Selection Distribution

Each token is selected proportionally to its weight:

```
Token 0 (weight=1): selected if random ∈ [0, 1)    → P = 1/9  = 11.1%
Token 1 (weight=3): selected if random ∈ [1, 4)    → P = 3/9  = 33.3%
Token 2 (weight=5): selected if random ∈ [4, 9)    → P = 5/9  = 55.6%
Token 3 (weight=3): excluded (beyond boundary)
Token 4 (weight=7): excluded
Token 5 (weight=1): excluded
```

## Edge Cases

### Case 1: All tokens have equal weight

```
Weights: [1, 1, 1, 1]
Cumsum:  [1, 2, 3, 4]
p = 0.5, threshold = 2
Boundary idx = 1, boundary_sum = 2
Random from [0, 2): each token 0,1 has 50% probability
```

### Case 2: One dominant token

```
Weights: [1000, 1, 1, 1]
Cumsum:  [1000, 1001, 1002, 1003]
p = 0.9, threshold = 902
Boundary idx = 0, boundary_sum = 1000
Random from [0, 1000): token 0 selected ~100% of time
```

### Case 3: p = 1.0 (include all tokens)

```
Weights: [1, 3, 5, 3, 7, 1]
Cumsum:  [1, 4, 9, 12, 19, 20]
p = 1.0, threshold = 20
Boundary idx = 5 (last token), boundary_sum = 20
Random from [0, 20): all tokens included proportionally
```

### Case 4: Sum approaches INT32_MAX

The implementation scales weights to use ~2^30 (1 billion) as max sum:

```
MAX_SUM = 2^30 = 1,073,741,824

For k=1024 tokens:
  Max weight per token ≈ 2^30 / 1024 = 1,048,576
  This prevents overflow during cumsum operations
```

## Two-Pass Vectorized Comparison

### Why Two-Pass Instead of Binary Search?

While binary search has O(log k) complexity, the two-pass vectorized approach is more efficient on TPU:

1. **Vectorization**: Single comparison across all k elements exploits SIMD
2. **No indexing overhead**: Binary search requires array indexing in the loop
3. **TPU-optimized**: Reduction operations (sum) are highly optimized on TPU
4. **Memory efficiency**: Single pass through cumsum array

### Two-Pass Implementation

```python
# Pass 1: Compute threshold
threshold = p * total_sum

# Pass 2: Vectorized comparison + reduction
below_threshold = cumsum < threshold  # Shape (batch, k) - vectorized
boundary_idx = below_threshold.sum(axis=-1)  # Shape (batch,) - reduction
```

### Finding Top-P Boundary

```python
def find_top_p_boundary(cumsum, total, p):
  """
  Find idx where cumsum[idx] >= threshold using two passes.

  Pass 1: Compute threshold = p * total
  Pass 2: Count elements where cumsum < threshold
  """
  threshold = (p * total).astype(int32)
  below_threshold = cumsum < threshold  # Vectorized comparison
  boundary_idx = below_threshold.sum(axis=-1)  # Efficient reduction
  return boundary_idx
```

### Token Selection

```python
def sample_token(cumsum, random_int):
  """
  Find token containing random_int using two passes.

  Pass 1: random_int is the threshold (already computed)
  Pass 2: Count elements where cumsum <= random_int
  """
  exceeds = cumsum > random_int  # Vectorized comparison
  token_idx = (~exceeds).sum(axis=-1)  # Count non-exceeding positions
  return token_idx
```

## Performance Characteristics

### Memory

- **Weights**: O(batch_size * k) int32
- **Cumsum**: O(batch_size * k) int32
- **Comparison mask**: O(batch_size * k) bool (temporary)

Total: Similar memory to float32 approach, but with better precision.

### Computation

- **Logits → int32**: O(batch_size * k) exp + scale
- **Cumsum**: O(batch_size * k) parallel scan
- **Boundary search**: O(batch_size * k) vectorized comparison + O(batch_size) reduction
- **Token search**: O(batch_size * k) vectorized comparison + O(batch_size) reduction

Total: O(batch_size * k) with excellent TPU utilization.

### Vectorization Benefits

Two-pass approach enables:
- Single vectorized comparison across all k elements (exploits SIMD)
- Highly optimized reduction operations on TPU
- No repeated array indexing overhead
- Predictable memory access pattern (sequential)

## API Reference

### Main Functions

```python
def logits_to_int32_weights(logits, max_sum=2**30):
  """Convert logits to int32 weights scaled to prevent overflow."""

def int32_cumsum(weights, axis=-1):
  """Compute cumulative sum in int32."""

def find_top_p_boundary_int32(cumsum_weights, total_weights, p):
  """Two-pass comparison to find top-p boundary index."""

def sample_token_from_int32_cumsum(cumsum_weights, random_int):
  """Two-pass comparison to select token from cumulative distribution."""

def top_p_and_sample_int32(logits, indices, rng_key, top_p):
  """Complete top-p sampling pipeline using int32 arithmetic."""

def int32_bsearch(batch_shape, predicate):
  """Helper: Binary search over int32 (kept for potential future use)."""
```

## Integration with Existing Code

The int32 sampling can be integrated into existing sampling pipelines:

```python
from tallax.tax.int32_sampling import top_p_and_sample_int32

# Traditional approach
# sampled = top_p_and_sample(logits, indices, key, p)

# Int32 approach
sampled = top_p_and_sample_int32(logits, indices, key, p)
```

Or use the int32 top-p mask in the existing pipeline:

```python
from tallax.vllm.top_p_and_sample import top_p_mask_int32

# In top_p_and_sample_arrays:
topp_logits = top_p_mask_int32(
  topk_logits=topk_logits,
  p=top_p,
  replace_val=replace_val,
  axis=0
)
```

## Testing

Comprehensive tests cover:

- Basic functionality with example `[1,3,5,3,7,1]`
- Edge cases (p=0, p=1, equal weights, dominant token)
- Overflow prevention (large vocabularies, extreme logits)
- Numerical stability (very small/large logits)
- Distribution properties (uniformity, proportionality)

Run tests:
```bash
pytest tests/int32_sampling_test.py -v
```

## Future Work

- [ ] Support for other dtypes (int64 for very large vocabularies)
- [ ] Optimize reduction operations for specific k values (e.g., k=128, k=1024)
- [ ] Integration with Pallas kernels for fused execution
- [ ] Benchmark against float32 approach on different hardware (TPU v4, v5)
