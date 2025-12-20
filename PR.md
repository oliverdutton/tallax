# Comparison: Tallax Divide & Filter TopK vs. TPU-KNN

## Executive Summary

*   **Tallax (Divide and Filter):** An **exact** Top-K algorithm. It uses a "Deep & Narrow" strategy, maintaining a sorted list of candidates per hardware lane and adaptively increasing the search depth ($m$) until it can mathematically prove that the global Top-K have been found.
*   **TPU-KNN (Paper):** An **approximate** Top-K algorithm. It uses a "Shallow & Wide" strategy, mapping items to a large number of bins ($L$) and keeping only the single best item per bin. It relies on probability theory to ensure high recall but does not guarantee exactness.

## Pros and Cons

### Tallax: Divide and Filter TopK
**Pros:**
*   **Exactness:** Guarantees 100% recall (returns the true Top-K) by verifying a convergence criterion.
*   **Robustness:** The adaptive schedule automatically handles skewed data distributions (e.g., if all top values map to the same lane) by increasing the search depth.
*   **Usability:** No tuning required. Users simply request `k`, and the algorithm determines the necessary parameters.
*   **Dynamic K:** Natively supports different `k` values for each token in the batch.

**Cons:**
*   **Variable Latency:** The runtime is data-dependent. "Hard" inputs (high collision rates) require deeper searches and take longer.
*   **Complexity:** The "sinking sort" and iterative control flow consume more instruction bandwidth than a simple reduction.

### TPU-KNN (Paper)
**Pros:**
*   **Peak Performance:** Designed to maximize FLOP/s with a fixed, predictable latency (Single-pass).
*   **Simplicity:** The core "PartialReduce" kernel is extremely lightweight (hash -> load -> compare -> store).
*   **Efficiency:** Excellent for scenarios where 95-99% recall is acceptable and speed is the priority.

**Cons:**
*   **Approximate:** Does not guarantee finding the true Top-K. Collisions (e.g., the 1st and 2nd best items mapping to the same bin) result in data loss.
*   **Tuning Required:** Users must select an output size $L$ significantly larger than $K$ (e.g., $L \approx 10K$) to achieve high recall, increasing memory bandwidth.
*   **Data Sensitivity:** Performance/Recall depends on the randomness of input indices. Sorted or structured data can cause catastrophic collisions without index permutation.

## Technical Differences

| Feature | Tallax (Divide & Filter) | TPU-KNN (Paper) |
| :--- | :--- | :--- |
| **Core Strategy** | **Deep & Narrow**<br>Fixed width (128 HW lanes)<br>Variable depth (Top-$M$ list) | **Shallow & Wide**<br>Variable width ($L$ bins)<br>Fixed depth (Top-1 item) |
| **State per Bin** | **Sorted List (Size $M$)**<br>Uses a "sinking sort" to maintain the top $M$ candidates seen so far in this lane. | **Single Value (Max)**<br>Stores only the current maximum value for this bin. |
| **Binning Logic** | **Modulo Sharding**<br>Items are split deterministically across hardware lanes (e.g., `idx % 128`). | **Spatial/Hash Binning**<br>Items are mapped to logical bins based on index blocks or hashing. |
| **Collision Handling** | **Adaptive (Iterative)**<br>If the Top-K are clustered in one lane, the algorithm increases $M$ and re-runs or merges bins until they are captured. | **Probabilistic (One-shot)**<br>Relies on having many more bins than $K$ ($L \gg K$) so that the probability of collisions is statistically low. |
| **Convergence** | **Provable (Min-Max)**<br>Calculates a pivot (max of the worst values per lane) to prove the global Top-K are strictly bounded within the candidate set. | **Statistical (Birthday Paradox)**<br>Uses the "Balls in Bins" theory to estimate the expected recall based on the ratio of bins ($L$) to items ($K$). |
