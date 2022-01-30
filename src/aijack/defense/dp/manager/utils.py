import math

import numpy as np
from scipy import special


def _log_add(logx, logy):
    """Add two numbers in the log space."""
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx, logy):
    """Subtract two numbers in the log space. Answer must be non-negative."""
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx


def _log_erfc(x):
    """Compute log(erfc(x)) with high accuracy for large x."""
    try:
        return math.log(2) + special.log_ndtr(-x * 2**0.5)
    except NameError:
        # If log_ndtr is not available, approximate as follows:
        r = special.erfc(x)
        if r == 0.0:
            # Using the Laurent series at infinity for the tail of the erfc function:
            #     erfc(x) ~ exp(-x^2-.5/x^2+.625/x^4)/(x*pi^.5)
            # To verify in Mathematica:
            #     Series[Log[Erfc[x]] + Log[x] + Log[Pi]/2 + x^2, {x, Infinity, 6}]
            return (
                -math.log(math.pi) / 2
                - math.log(x)
                - x**2
                - 0.5 * x**-2
                + 0.625 * x**-4
                - 37.0 / 24.0 * x**-6
                + 353.0 / 64.0 * x**-8
            )
        else:
            return math.log(r)
