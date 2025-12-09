"""Tallax utilities module.

Public API for utility functions and constants.
"""

# Import from _src.utils using JAX's pattern
# The "as <name>" syntax is required for proper re-export
from tallax._src.utils import NUM_LANES as NUM_LANES
from tallax._src.utils import NUM_SUBLANES as NUM_SUBLANES
from tallax._src.utils import is_cpu_platform as is_cpu_platform
from tallax._src.utils import log2 as log2
from tallax._src.utils import max_int as max_int
from tallax._src.utils import all_concrete_ints as all_concrete_ints
from tallax._src.utils import get_dtype_info as get_dtype_info
from tallax._src.utils import pad as pad
from tallax._src.utils import standardize as standardize
from tallax._src.utils import is_32bit as is_32bit
from tallax._src.utils import to_32bit_dtype as to_32bit_dtype
from tallax._src.utils import same_shape_dtype as same_shape_dtype
from tallax._src.utils import canonicalize_operand as canonicalize_operand
from tallax._src.utils import float_to_sortable_int as float_to_sortable_int
from tallax._src.utils import sortable_int_to_float as sortable_int_to_float
from tallax._src.utils import pack_bf16_u16_to_i32 as pack_bf16_u16_to_i32
from tallax._src.utils import unpack_bf16_u16_from_i32 as unpack_bf16_u16_from_i32
from tallax._src.utils import split_array_to_tiles as split_array_to_tiles
from tallax._src.utils import join_tiles_to_array as join_tiles_to_array
from tallax._src.utils import iota_tile as iota_tile
from tallax._src.utils import create_bit_indicator as create_bit_indicator
from tallax._src.utils import convert_to_sublane_sort_format as convert_to_sublane_sort_format
from tallax._src.utils import convert_from_sublane_sort_format as convert_from_sublane_sort_format
from tallax._src.utils import unrolled_fori_loop as unrolled_fori_loop
from tallax._src.utils import transpose_list_of_lists as transpose_list_of_lists

__all__ = [
    "NUM_LANES",
    "NUM_SUBLANES",
    "is_cpu_platform",
    "log2",
    "max_int",
    "all_concrete_ints",
    "get_dtype_info",
    "pad",
    "standardize",
    "is_32bit",
    "to_32bit_dtype",
    "same_shape_dtype",
    "canonicalize_operand",
    "float_to_sortable_int",
    "sortable_int_to_float",
    "pack_bf16_u16_to_i32",
    "unpack_bf16_u16_from_i32",
    "split_array_to_tiles",
    "join_tiles_to_array",
    "iota_tile",
    "create_bit_indicator",
    "convert_to_sublane_sort_format",
    "convert_from_sublane_sort_format",
    "unrolled_fori_loop",
    "transpose_list_of_lists",
]
