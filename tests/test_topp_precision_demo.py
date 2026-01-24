"""Demonstrate precision differences by exploiting softmax+cumsum ordering.

The key difference between implementations:
- f32: softmax(logits) -> cumsum in f32 -> compare to threshold
- i32: softmax(logits) -> scale to i32 -> cumsum in i32 -> scale back -> compare

We need logits where the softmax probabilities, when accumulated in different
precisions, cross the top_p threshold at different points.
