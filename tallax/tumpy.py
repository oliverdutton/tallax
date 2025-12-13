import jax.numpy as jnp
from tallax._src.sort import sort as lax_sort_pallas
from tallax._src.gather import take_along_axis as _take_along_axis
from tallax._src.utils import is_cpu_platform

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

    Note:
        Requires non-empty, non-scalar array. Only supports axis=-1 or last dimension.
    """
    # Shape validations
    if a.ndim == 0:
        raise ValueError("Cannot sort scalar arrays")

    if a.size == 0:
        raise ValueError("Cannot sort empty arrays")

    if axis is not None:
        ndim = a.ndim
        canonical_axis = axis if axis >= 0 else axis + ndim
        if canonical_axis != ndim - 1:
            raise ValueError(
                f"tumpy only supports sorting along the last axis. "
                f"Got axis={axis} (canonical: {canonical_axis}) for {ndim}D array"
            )

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

    Note:
        Requires non-empty, non-scalar array. Only supports axis=-1 or last dimension.
    """
    # Shape validations
    if a.ndim == 0:
        raise ValueError("Cannot argsort scalar arrays")

    if a.size == 0:
        raise ValueError("Cannot argsort empty arrays")

    if axis is not None:
        ndim = a.ndim
        canonical_axis = axis if axis >= 0 else axis + ndim
        if canonical_axis != ndim - 1:
            raise ValueError(
                f"tumpy only supports sorting along the last axis. "
                f"Got axis={axis} (canonical: {canonical_axis}) for {ndim}D array"
            )

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

    Note:
        Requires arr and indices to have matching shapes except along axis dimension.
    """
    # Shape validations
    if arr.ndim != indices.ndim:
        raise ValueError(
            f"Arrays must have same number of dimensions, "
            f"got arr.ndim={arr.ndim} and indices.ndim={indices.ndim}"
        )

    # Validate axis
    if axis < -arr.ndim or axis >= arr.ndim:
        raise ValueError(
            f"Invalid axis {axis} for array with {arr.ndim} dimensions"
        )

    # Normalize axis to positive
    normalized_axis = axis if axis >= 0 else axis + arr.ndim

    # Check non-axis dimensions match
    for i in range(arr.ndim):
        if i != normalized_axis and arr.shape[i] != indices.shape[i]:
            raise ValueError(
                f"Non-axis dimension {i} must match: "
                f"arr.shape[{i}]={arr.shape[i]} != indices.shape[{i}]={indices.shape[i]}"
            )

    return _take_along_axis(arr, indices, axis, interpret=interpret)
