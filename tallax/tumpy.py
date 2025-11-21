import jax.numpy as jnp
from .tax.sort import sort as lax_sort_pallas
from .utils import is_cpu_platform

def sort(a, axis=-1, kind=None, order=None, stable=True, descending=False):
    """
    Sort an array along the last axis using Pallas-based bitonic sort.

    Args:
        a: Input array.
        axis: Axis along which to sort. Must be -1 or the last dimension.
              If None, the array is flattened before sorting.
        kind: Sort kind (ignored, for compatibility with numpy).
        order: Sort order. If 'descending', sorts in descending order.
        stable: Whether to use stable sort.
        descending: Whether to sort in descending order.

    Returns:
        Sorted array.
    """
    if order == 'descending':
        descending = True

    interpret = is_cpu_platform()

    if axis is None:
        # Flatten to 1D
        a_flat = a.ravel()
        # Sort 1D array
        # lax_sort_pallas expects [a] where a is 2D usually?
        # No, lax_sort_pallas logic:
        # operands, shape = canonicalize_operand(operand)
        # canonicalize_operand enforces 2D.
        # So we must reshape 1D to (1, size).

        size = a_flat.size
        a_reshaped = a_flat.reshape((1, size))

        sorted_arrays = lax_sort_pallas(
            [a_reshaped],
            num_keys=1,
            is_stable=stable,
            descending=descending,
            interpret=interpret,
            return_argsort=False
        )
        return sorted_arrays[0].reshape((size,))

    ndim = a.ndim
    canonical_axis = axis if axis >= 0 else axis + ndim

    if canonical_axis != ndim - 1:
        raise ValueError("tumpy only supports sorting along the last axis.")

    # Reshape to 2D: (product of batch dims, sort dim)
    original_shape = a.shape
    sort_dim_size = original_shape[-1]

    if ndim > 1:
        batch_dim_size = 1
        for dim in original_shape[:-1]:
            batch_dim_size *= dim
        a_reshaped = a.reshape((batch_dim_size, sort_dim_size))
    else:
        # 1D array -> (1, N)
        a_reshaped = a.reshape((1, sort_dim_size))

    sorted_arrays = lax_sort_pallas(
        [a_reshaped],
        num_keys=1,
        is_stable=stable,
        descending=descending,
        interpret=interpret,
        return_argsort=False
    )

    sorted_a = sorted_arrays[0]

    if ndim > 1:
        return sorted_a.reshape(original_shape)
    else:
        return sorted_a.reshape(original_shape)

def argsort(a, axis=-1, kind=None, order=None, stable=True, descending=False):
    """
    Return the indices that would sort an array along the last axis.

    Args:
        a: Input array.
        axis: Axis along which to sort. Must be -1 or the last dimension.
              If None, the array is flattened before sorting.
        kind: Sort kind (ignored, for compatibility with numpy).
        order: Sort order. If 'descending', sorts in descending order.
        stable: Whether to use stable sort.
        descending: Whether to sort in descending order.

    Returns:
        Array of indices.
    """
    if order == 'descending':
        descending = True

    interpret = is_cpu_platform()

    if axis is None:
        a_flat = a.ravel()
        size = a_flat.size
        a_reshaped = a_flat.reshape((1, size))

        results = lax_sort_pallas(
            [a_reshaped],
            num_keys=1,
            is_stable=stable,
            descending=descending,
            interpret=interpret,
            return_argsort=True
        )
        return results[-1].reshape((size,))

    ndim = a.ndim
    canonical_axis = axis if axis >= 0 else axis + ndim

    if canonical_axis != ndim - 1:
        raise ValueError("tumpy only supports sorting along the last axis.")

    original_shape = a.shape
    sort_dim_size = original_shape[-1]

    if ndim > 1:
        batch_dim_size = 1
        for dim in original_shape[:-1]:
            batch_dim_size *= dim
        a_reshaped = a.reshape((batch_dim_size, sort_dim_size))
    else:
        a_reshaped = a.reshape((1, sort_dim_size))

    results = lax_sort_pallas(
        [a_reshaped],
        num_keys=1,
        is_stable=stable,
        descending=descending,
        interpret=interpret,
        return_argsort=True
    )

    indices = results[-1]

    if ndim > 1:
        return indices.reshape(original_shape)
    else:
        return indices.reshape(original_shape)
