import jax.numpy as jnp
from tallax.tax.sort import sort as lax_sort_pallas
from tallax.tax.gather import take_along_axis as _take_along_axis
from tallax.utils import is_cpu_platform

def sort(a, axis=-1, kind=None, order=None, stable=True, descending=False, interpret=False):
    """
    Sort an array along the last axis using Pallas-based bitonic sort.

    Args:
        a: Input array.
        axis: Axis along which to sort. Must be -1 or the last dimension.
              If None, the array is flattened before sorting.
        kind: Sort kind (ignored, for compatibility with numpy).
        order: Sort order (ignored, for compatibility with numpy).
        stable: Whether to use stable sort.
        descending: Whether to sort in descending order.
        interpret: Whether to use interpreter mode (for testing).

    Returns:
        Sorted array.
    """
    if axis is None:
        a_work = a.ravel()
        target_shape = a_work.shape
        sort_dim_size = a_work.size
    else:
        ndim = a.ndim
        canonical_axis = axis if axis >= 0 else axis + ndim
        if canonical_axis != ndim - 1:
            raise ValueError("tumpy only supports sorting along the last axis.")

        a_work = a
        target_shape = a.shape
        sort_dim_size = a.shape[-1]

    # Reshape to (-1, sort_dim_size)
    # If 1D, reshapes to (1, sort_dim_size) which matches (1, size) logic.
    # If ND, reshapes to (batch, sort_dim_size).
    a_reshaped = a_work.reshape(-1, sort_dim_size)

    # lax_sort_pallas returns a tuple of sorted arrays
    (sorted_a,) = lax_sort_pallas(
        a_reshaped,
        num_keys=1,
        is_stable=stable,
        descending=descending,
        interpret=interpret,
        return_argsort=False
    )

    return sorted_a.reshape(target_shape)

def argsort(a, axis=-1, kind=None, order=None, stable=True, descending=False, interpret=False):
    """
    Return the indices that would sort an array along the last axis.

    Args:
        a: Input array.
        axis: Axis along which to sort. Must be -1 or the last dimension.
              If None, the array is flattened before sorting.
        kind: Sort kind (ignored, for compatibility with numpy).
        order: Sort order (ignored, for compatibility with numpy).
        stable: Whether to use stable sort.
        descending: Whether to sort in descending order.
        interpret: Whether to use interpreter mode (for testing).

    Returns:
        Array of indices.
    """
    if axis is None:
        a_work = a.ravel()
        target_shape = a_work.shape
        sort_dim_size = a_work.size
    else:
        ndim = a.ndim
        canonical_axis = axis if axis >= 0 else axis + ndim
        if canonical_axis != ndim - 1:
            raise ValueError("tumpy only supports sorting along the last axis.")

        a_work = a
        target_shape = a.shape
        sort_dim_size = a.shape[-1]

    a_reshaped = a_work.reshape(-1, sort_dim_size)

    # return_argsort=True returns (sorted_arrays..., indices)
    _, indices = lax_sort_pallas(
        a_reshaped,
        num_keys=1,
        is_stable=stable,
        descending=descending,
        interpret=interpret,
        return_argsort=True
    )

    return indices.reshape(target_shape)

def take_along_axis(arr, indices, axis, interpret=False):
    """
    Take values from the input array by matching 1d index and data slices.

    Args:
        arr: Input array.
        indices: Indices to take along each 1d slice of arr.
        axis: The axis along which to select values.
        interpret: Whether to use interpreter mode (for testing).

    Returns:
        Array with selected values.
    """
    return _take_along_axis(arr, indices, axis, interpret=interpret)
