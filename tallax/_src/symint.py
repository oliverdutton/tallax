"""SymInt: A wrapper around JAX tracers with static bound metadata.

This class enables static evaluation of comparisons and operations when bounds
allow concrete determination, reducing compilation overhead in Pallas kernels.
"""

import jax
import jax.numpy as jnp


class SymInt:
    """Integer wrapper with lower and upper bound metadata.

    Wraps a JAX tracer (typically from pl.loop) with compile-time known bounds,
    enabling static evaluation of comparisons and optimized code generation.

    Args:
        value: The actual JAX tracer or integer value
        lower_bound: Inclusive lower bound (compile-time constant)
        upper_bound: Inclusive upper bound (compile-time constant)

    Example:
        >>> @pl.loop(0, 10)
        >>> def loop_body(i):
        >>>     i_bounded = SymInt(i, lower_bound=0, upper_bound=9)
        >>>     # Can statically determine: i_bounded < 100 -> True
        >>>     if i_bounded < 100:  # Returns True, no dynamic check
        >>>         ...
    """

    def __init__(self, value, lower_bound: int, upper_bound: int):
        self.value = value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __repr__(self):
        return f"SymInt({self.value}, [{self.lower_bound}, {self.upper_bound}])"

    # Arithmetic operations - track bounds through operations

    def __add__(self, other):
        if isinstance(other, int):
            return SymInt(
                self.value + other,
                self.lower_bound + other,
                self.upper_bound + other
            )
        elif isinstance(other, SymInt):
            return SymInt(
                self.value + other.value,
                self.lower_bound + other.lower_bound,
                self.upper_bound + other.upper_bound
            )
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, int):
            return SymInt(
                self.value - other,
                self.lower_bound - other,
                self.upper_bound - other
            )
        elif isinstance(other, SymInt):
            return SymInt(
                self.value - other.value,
                self.lower_bound - other.upper_bound,  # Note: reverse for subtraction
                self.upper_bound - other.lower_bound
            )
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, int):
            return SymInt(
                other - self.value,
                other - self.upper_bound,  # Note: reverse for subtraction
                other - self.lower_bound
            )
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, int):
            if other >= 0:
                return SymInt(
                    self.value * other,
                    self.lower_bound * other,
                    self.upper_bound * other
                )
            else:
                # Negative multiplier reverses bounds
                return SymInt(
                    self.value * other,
                    self.upper_bound * other,
                    self.lower_bound * other
                )
        elif isinstance(other, SymInt):
            # Conservative bounds for general case
            products = [
                self.lower_bound * other.lower_bound,
                self.lower_bound * other.upper_bound,
                self.upper_bound * other.lower_bound,
                self.upper_bound * other.upper_bound
            ]
            return SymInt(
                self.value * other.value,
                min(products),
                max(products)
            )
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __floordiv__(self, other):
        if isinstance(other, int) and other > 0:
            return SymInt(
                self.value // other,
                self.lower_bound // other,
                self.upper_bound // other
            )
        elif isinstance(other, int) and other < 0:
            # Negative divisor reverses bounds
            return SymInt(
                self.value // other,
                self.upper_bound // other,
                self.lower_bound // other
            )
        return NotImplemented

    def __mod__(self, other):
        if isinstance(other, int) and other > 0:
            # For positive modulus, result is always in [0, other-1]
            # But if we know more about the input bounds, we can be tighter
            if self.lower_bound >= 0:
                # Non-negative input
                if self.upper_bound < other:
                    # No wrapping occurs
                    return SymInt(
                        self.value % other,
                        self.lower_bound,
                        self.upper_bound
                    )
            # General case: could be any value in [0, other-1]
            return SymInt(
                self.value % other,
                0,
                other - 1
            )
        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, int):
            if other == 0:
                # x^0 = 1
                return SymInt(1, 1, 1)
            elif other > 0:
                if self.lower_bound >= 0:
                    # Non-negative base
                    return SymInt(
                        self.value ** other,
                        self.lower_bound ** other,
                        self.upper_bound ** other
                    )
                else:
                    # Mixed sign bounds - conservative estimate
                    candidates = [
                        abs(self.lower_bound) ** other,
                        abs(self.upper_bound) ** other
                    ]
                    max_abs = max(candidates)
                    if other % 2 == 0:
                        # Even power: always non-negative
                        return SymInt(
                            self.value ** other,
                            0,
                            max_abs
                        )
                    else:
                        # Odd power: maintains sign
                        return SymInt(
                            self.value ** other,
                            -(max_abs),
                            max_abs
                        )
        elif isinstance(other, SymInt):
            # For 2^stage where stage is SymInt
            # This is the common case in bitonic sort
            if isinstance(self.value, int) and self.value == 2:
                return SymInt(
                    2 ** other.value,
                    2 ** other.lower_bound,
                    2 ** other.upper_bound
                )
        return NotImplemented

    def __rpow__(self, other):
        # For expressions like 2**stage where other=2, self=stage
        if isinstance(other, int) and other >= 0:
            if self.lower_bound >= 0:
                return SymInt(
                    other ** self.value,
                    other ** self.lower_bound,
                    other ** self.upper_bound
                )
        return NotImplemented

    # Comparison operations - return concrete values when bounds allow

    def __lt__(self, other):
        if isinstance(other, int):
            if self.upper_bound < other:
                # All possible values are less than other
                return True
            elif self.lower_bound >= other:
                # All possible values are >= other
                return False
            else:
                # Dynamic check needed
                return self.value < other
        elif isinstance(other, SymInt):
            if self.upper_bound < other.lower_bound:
                return True
            elif self.lower_bound >= other.upper_bound:
                return False
            else:
                return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, int):
            if self.upper_bound <= other:
                return True
            elif self.lower_bound > other:
                return False
            else:
                return self.value <= other
        elif isinstance(other, SymInt):
            if self.upper_bound <= other.lower_bound:
                return True
            elif self.lower_bound > other.upper_bound:
                return False
            else:
                return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, int):
            if self.lower_bound > other:
                return True
            elif self.upper_bound <= other:
                return False
            else:
                return self.value > other
        elif isinstance(other, SymInt):
            if self.lower_bound > other.upper_bound:
                return True
            elif self.upper_bound <= other.lower_bound:
                return False
            else:
                return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, int):
            if self.lower_bound >= other:
                return True
            elif self.upper_bound < other:
                return False
            else:
                return self.value >= other
        elif isinstance(other, SymInt):
            if self.lower_bound >= other.upper_bound:
                return True
            elif self.upper_bound < other.lower_bound:
                return False
            else:
                return self.value >= other.value
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, int):
            if self.lower_bound == self.upper_bound == other:
                # Value is statically known to equal other
                return True
            elif other < self.lower_bound or other > self.upper_bound:
                # Value is statically known to not equal other
                return False
            else:
                # Dynamic check needed
                return self.value == other
        elif isinstance(other, SymInt):
            if (self.lower_bound == self.upper_bound ==
                other.lower_bound == other.upper_bound):
                # Both are statically known constants
                return self.lower_bound == other.lower_bound
            else:
                return self.value == other.value
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if isinstance(result, bool):
            return not result
        else:
            # Dynamic expression
            return self.value != (other if isinstance(other, int) else other.value)

    # Bitwise operations

    def __and__(self, other):
        """Bitwise AND - conservative bounds."""
        if isinstance(other, int):
            # x & other <= min(x, other)
            return SymInt(
                self.value & other,
                0,  # Conservative lower bound
                min(self.upper_bound, other)
            )
        return NotImplemented

    def __or__(self, other):
        """Bitwise OR - conservative bounds."""
        if isinstance(other, int):
            # x | other >= max(x, other)
            return SymInt(
                self.value | other,
                max(self.lower_bound, other),
                self.upper_bound | other  # Conservative
            )
        return NotImplemented

    def __xor__(self, other):
        """Bitwise XOR - conservative bounds."""
        if isinstance(other, int):
            # Very conservative
            return SymInt(
                self.value ^ other,
                0,
                max(self.upper_bound, other) * 2  # Very conservative
            )
        return NotImplemented

    def __lshift__(self, other):
        """Left bitshift: x << other."""
        if isinstance(other, int) and other >= 0:
            return SymInt(
                self.value << other,
                self.lower_bound << other,
                self.upper_bound << other
            )
        return NotImplemented

    def __rshift__(self, other):
        """Right bitshift: x >> other."""
        if isinstance(other, int) and other >= 0:
            return SymInt(
                self.value >> other,
                self.lower_bound >> other,
                self.upper_bound >> other
            )
        return NotImplemented

    # Reverse bitwise operations (for when SymInt is on the right side)

    def __rand__(self, other):
        """Reverse bitwise AND: other & self."""
        if isinstance(other, int):
            return SymInt(
                other & self.value,
                0,  # Conservative
                min(other, self.upper_bound)
            )
        return NotImplemented

    def __ror__(self, other):
        """Reverse bitwise OR: other | self."""
        if isinstance(other, int):
            return SymInt(
                other | self.value,
                max(other, self.lower_bound),
                other | self.upper_bound  # Conservative
            )
        return NotImplemented

    def __rxor__(self, other):
        """Reverse bitwise XOR: other ^ self."""
        if isinstance(other, int):
            return SymInt(
                other ^ self.value,
                0,
                max(other, self.upper_bound) * 2  # Conservative
            )
        return NotImplemented

    def __rlshift__(self, other):
        """Reverse left bitshift: other << self.

        This is tricky because the result depends exponentially on self.
        For safety, we use the actual tracer value for computation.
        """
        if isinstance(other, int):
            if self.lower_bound == self.upper_bound:
                # Concrete shift amount
                shift = self.lower_bound
                return SymInt(
                    other << self.value,
                    other << shift,
                    other << shift
                )
            else:
                # Dynamic shift - bounds are exponential
                return SymInt(
                    other << self.value,
                    other << self.lower_bound,
                    other << self.upper_bound
                )
        # For JAX tracer on the left
        return other << self.value

    def __rrshift__(self, other):
        """Reverse right bitshift: other >> self.

        Shifts the other value right by self amount.
        """
        if isinstance(other, int):
            if self.lower_bound == self.upper_bound:
                # Concrete shift amount
                shift = self.lower_bound
                return SymInt(
                    other >> self.value,
                    other >> shift,
                    other >> shift
                )
            else:
                # Dynamic shift
                return SymInt(
                    other >> self.value,
                    other >> self.upper_bound,  # Right shift: larger shift = smaller result
                    other >> self.lower_bound
                )
        # For JAX tracer on the left
        return other >> self.value

    # Allow use as integer index

    def __index__(self):
        """Convert to integer index for array indexing."""
        if hasattr(self.value, '__index__'):
            return self.value.__index__()
        else:
            return int(self.value)

    def __int__(self):
        """Convert to integer."""
        return int(self.value)

    def __hash__(self):
        """Make hashable for use in dicts/sets."""
        # We can't hash the JAX tracer, so we use bounds
        return hash((self.lower_bound, self.upper_bound))
