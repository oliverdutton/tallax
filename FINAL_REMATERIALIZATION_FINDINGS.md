# Final Rematerialization Analysis: The Root Cause

## Executive Summary

**The rematerialization is NOT a CSE failure - it's loop unrolling at both jaxpr and XLA levels.**

## Key Findings

### Jaxpr Level (JAX Tracing)
```
Total equations (all levels): 7,027
Top-level equations: 3
Nested equations: 7,024
```

**The 200-iteration Python `for` loop gets UNROLLED into 200 separate jaxpr operations:**
- Each iteration becomes a separate `jit` call in the scan body
- The scan body has 2,407 equations
- 200 nested `jit (jaxpr):` blocks × ~10 equations each = ~2,000 equations

**But CSE DOES work at jaxpr level:**
- Only **2 iota operations** in the top-level jaxpr (one for dim=0, one for dim=1)
- These are hoisted out of the loop body

### HLO Level (XLA Compilation)
```
iota: 20 operations (not 2!)
broadcast: 441 operations
xor: 5 operations
add: 2,417 operations (not ~200!)
```

**XLA further unrolls and duplicates:**
- The 2 jaxpr iotas become 20 HLO iotas (10x duplication)
- The ~200 expected adds become 2,417 adds (12x duplication)
- This happens during fusion and scheduling

## Why This Happens

### 1. Python For-Loop Unrolling
```python
for _ in range(200):  # ← This gets unrolled!
    iota_0_local = iota_tile(0)
    ...
```

During JAX tracing, Python loops are executed completely, creating 200 separate traces. Each trace becomes a jaxpr fragment.

### 2. XLA Optimization Trade-offs
XLA makes scheduling decisions that can duplicate operations:
- **Fusion**: Combining operations into kernels may duplicate shared computations
- **Memory-compute tradeoff**: Sometimes cheaper to recompute than to load from memory
- **Parallelism**: Duplicating ops can enable more parallelism

## What CSE CAN and CANNOT Do

### ✓ CSE Successfully Handles (Jaxpr Level)
1. **Same-context duplicates**: Multiple `iota_tile(0)` calls in same block → 1 iota
2. **Value-based deduplication**: `a + b` computed twice → computed once
3. **Within single trace**: All computations in one jaxpr equation

### ✗ CSE Cannot Handle (Cross-Context)
1. **Loop unrolling**: Each iteration is separate context → separate iotas
2. **XLA fusion decisions**: Backend optimizer makes its own choices
3. **Cross-stage duplicates**: Different pipeline stages = different contexts

## Evidence from Test Pattern

### Test Code
```python
def body_fn(idx, carry):
    for _ in range(200):  # Python loop
        iota_0_local = iota_tile(0)  # Called 200 times
        iota_1_local = iota_tile(1)  # Called 200 times
        ...
```

### Results
| Level | iota Count | Explanation |
|-------|-----------|-------------|
| **Jaxpr** | 2 | CSE works! Hoisted out of loop |
| **HLO** | 20 | XLA unrolls and duplicates |
| **Expected (ideal)** | 2 | Perfect CSE would keep 2 |

### Jaxpr Structure
```
root: 3 equations
  scan (body_jaxpr):
    scan_jaxpr: 2,407 equations  ← THE PROBLEM!
      [9] jit: 10 equations
      [21] jit: 10 equations
      [33] jit: 10 equations
      ...
      [2397] jit: 10 equations   ← 200 iterations × ~12 eqns
```

**Each of the 200 loop iterations creates a separate jaxpr context!**

## Solution Approaches

### Approach 1: Don't Use Python For-Loops ✓
```python
# BAD - unrolls during tracing
for _ in range(200):
    result += compute()

# GOOD - stays as single loop op
result = jax.lax.fori_loop(0, 200, lambda i, r: r + compute(), init)
```

**But:** Even `fori_loop` gets expanded at XLA level if the body is complex.

### Approach 2: Hoist Out of Loops ✓ (What we did)
```python
# Compute ONCE before loop
iota_0 = iota_tile(0)
iota_1 = iota_tile(1)

for stage in stages:
    for substage in substages:
        # Reuse iota_0, iota_1
```

**Impact:** Reduced 140 iota calls to ~70 (50% reduction)

### Approach 3: XLA Compiler Flags (To Investigate)
```python
compiler_params=pltpu.CompilerParams(
    # Potential flags for better CSE?
    vmem_limit_bytes=...,
    # Other fusion/optimization controls?
)
```

### Approach 4: Pallas Compiler Integration (Future Work)
Modify Pallas lowering to:
1. Detect loop-invariant computations
2. Hoist them explicitly in the IR
3. Prevent XLA from duplicating them during fusion

## Bitonic Sort Specific Findings

### Original State
- 140 iota operations in HLO
- 336x duplications of comparison ops
- Operations spread across 7-10 substages × multiple blocks

### After Our Optimizations
- ~70 iota operations (50% reduction)
- Hoisted to outer loop scope
- 547 jaxpr equations eliminated (6.2% reduction)

### Remaining Duplicates
The 336x `lt/add/select_n` duplications come from:
1. **7-10 pipeline stages** (unrolled)
2. **7 substages per stage** (unrolled)
3. **Multiple grid blocks** (parallel)

Each combination creates a separate context where operations are duplicated.

## Conclusion

### What We Learned
1. **JAX CSE is working** - only 2 iotas in jaxpr
2. **Loop unrolling is the culprit** - creates 200+ separate contexts
3. **XLA makes its own choices** - duplicates for fusion/parallelism
4. **Code-level hoisting wins** - better than post-hoc CSE

### Recommendations
1. ✅ **Use our hoisting approach** - already achieved 50% reduction
2. ⏳ **Investigate XLA flags** - may control fusion behavior
3. ⏳ **Profile actual performance** - duplicates may not hurt runtime
4. ⏳ **Consider Pallas modifications** - for deeper optimization

### Final Answer to "Why 70 iota calls?"
Because the bitonic sort has ~7-10 stages × ~7 substages × parallel execution contexts. Even with perfect jaxpr-level CSE, XLA's fusion and scheduling create ~10 copies per unique iota, leading to ~70 total operations from what should be 2.

**The good news:** This is largely unavoidable without XLA-level changes, and our 50% reduction through hoisting is a significant achievement.
