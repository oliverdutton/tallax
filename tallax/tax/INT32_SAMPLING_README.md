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

Step 3: Find top-p boundary using binary search
p = 0.5, total = 1234803097
threshold = 0.5 * 1234803097 = 617401548
Binary search finds idx where cumsum >= threshold → idx = 2
boundary_sum = cumsum[2] = 536870912

Step 4: Sample random int32 from [0, boundary_sum)
random_int = 123456789  (uniformly sampled)

Step 5: Binary search to find token
Find idx where cumsum[idx-1] <= random_int < cumsum[idx]
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
  Threshold = 20 * 0.5 = 10
  Binary search finds boundary_idx = 2 (cumsum[2]=9 < 10, cumsum[3]=12 >= 10)
  Boundary sum = 9

Sampling:
  Random int: r = 7 (from [0, 9))

  Binary search for token:
    cumsum[0] = 1 <= 7?  Yes
    cumsum[1] = 4 <= 7?  Yes
    cumsum[2] = 9 <= 7?  No → Token is 2

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

## Binary Search for Latency Hiding

### Why Binary Search?

On TPU, memory operations have high latency. Binary search provides:

1. **Latency hiding**: Parallel memory fetches during search
2. **O(log k) vs O(k)**: Better complexity for large k
3. **Deterministic behavior**: Same number of operations regardless of data

### Binary Search Implementation

```python
def int32_bsearch(batch_shape, predicate):
  """
  Search for largest int32 where predicate is False.

  Uses bit-by-bit binary search from MSB to LSB.
  """
  current_bits = zeros(batch_shape, int32)

  # Special handling for sign bit (bit 31)
  midpoint = current_bits
  if predicate(midpoint):
    current_bits |= (1 << 31)

  # Search remaining bits (30 down to 0)
  for bit_index in range(30, -1, -1):
    midpoint = current_bits | (1 << bit_index)
    if not predicate(midpoint):
      current_bits |= (1 << bit_index)

  return current_bits
```

### Finding Top-P Boundary

```python
def find_boundary(cumsum, threshold):
  """Find idx where cumsum[idx] >= threshold"""

  def predicate(idx):
    return cumsum[idx] >= threshold

  # Returns largest idx where cumsum[idx] < threshold
  boundary_idx = int32_bsearch(batch_shape, predicate)
  return boundary_idx
```

### Token Selection

```python
def sample_token(cumsum, random_int):
  """Find token containing random_int"""

  def predicate(idx):
    return cumsum[idx] > random_int

  # Returns first idx where cumsum[idx] > random_int
  token_idx = int32_bsearch(batch_shape, predicate) + 1
  return token_idx
```

## Performance Characteristics

### Memory

- **Weights**: O(batch_size * k) int32
- **Cumsum**: O(batch_size * k) int32
- **Scratch**: O(batch_size) int32 for binary search

Total: ~3x the memory of float32 approach, but no precision loss.

### Computation

- **Logits → int32**: O(batch_size * k) exp + scale
- **Cumsum**: O(batch_size * k) parallel scan
- **Boundary search**: O(batch_size * 32) binary search iterations
- **Token search**: O(batch_size * 32) binary search iterations

Total: O(batch_size * k) with excellent TPU utilization.

### Latency Hiding

Binary search enables:
- Parallel memory fetches across batches
- Predictable memory access patterns
- Efficient use of TPU's memory hierarchy

## API Reference

### Main Functions

```python
def logits_to_int32_weights(logits, max_sum=2**30):
  """Convert logits to int32 weights scaled to prevent overflow."""

def int32_cumsum(weights, axis=-1):
  """Compute cumulative sum in int32."""

def find_top_p_boundary_int32(cumsum_weights, total_weights, p):
  """Binary search to find top-p boundary index."""

def sample_token_from_int32_cumsum(cumsum_weights, random_int):
  """Binary search to select token from cumulative distribution."""

def top_p_and_sample_int32(logits, indices, rng_key, top_p):
  """Complete top-p sampling pipeline using int32 arithmetic."""
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

- [ ] Optimize binary search for specific k values (e.g., k=128)
- [ ] Support for other dtypes (int64 for very large vocabularies)
- [ ] Vectorized binary search for multiple p values
- [ ] Integration with Pallas kernels for fused execution
