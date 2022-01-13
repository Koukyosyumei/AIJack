import numpy as np


def _ternary_search(f, left, right, orders, precision, max_iteration=1000):
    i = 0
    while i < max_iteration:
        if abs(right - left) < precision:
            return (left + right) / 2

        left_third = left + (right - left) / 3
        right_third = right - (right - left) / 3

        if f(left_third) < f(right_third):
            right = right_third
        else:
            left = left_third

        i += 1
    return (left + right) / 2


def _ternary_search_int(f, left, right, orders, precision, max_iteration=1000):
    lo = left
    hi = right

    i = 0
    while hi - lo > 1 and i < max_iteration:
        mid = (hi + lo) >> 1
        if f(mid) > f(mid + 1):
            hi = mid
        else:
            lo = mid
        i += 1
    return lo + 1


def _bisection_search(f, lam_min, lam_max, orders, precision, max_iteration=1000):
    i = 0
    lam = lam_min
    while lam <= lam_max and i < max_iteration:
        val = f(lam)
        val_prev = f(lam - 1)

        if val > val_prev:
            break

        lam *= 2
        i += 1

    optim_lam = _greedy_search(
        f, int(lam / 2), lam, None, precision, max_iteration=max_iteration
    )
    return optim_lam


def _greedy_search(f, lam_min, lam_max, orders, precision, max_iteration=1000):
    min_val = np.inf
    optim_lam = lam_min
    orders = range(lam_min, lam_max + 1) if orders is None else orders
    for lam in orders:
        val = f(lam)
        if min_val > val:
            optim_lam = lam
            min_val = val
    return optim_lam
