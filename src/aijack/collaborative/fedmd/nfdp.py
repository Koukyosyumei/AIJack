import math


def get_epsilon_of_fedmd_nfdp(n, k, replacement=True):
    """Return epsilon of FedMD-NFDP

    Args:
        n (int): training set size
        k (int): sampling size
        replacement (bool, optional): sampling w/o replacement. Defaults to True.

    Returns:
        float: epsilon of FedMD-NFDP
    """
    if replacement:
        return k * math.log((n + 1) / n)
    else:
        return math.log((n + 1) / (n + 1 - k))


def get_delta_of_fedmd_nfdp(n, k, replacement=True):
    """Return delta of FedMD-NFDP

    Args:
        n (int): training set size
        k (int): sampling size
        replacement (bool, optional): sampling w/o replacement. Defaults to True.

    Returns:
        float: delta of FedMD-NFDP
    """
    if replacement:
        return 1 - ((n - 1) / n) ** k
    else:
        return k / n


def get_k_of_fedmd_nfdp(epsilon, n, replacement=True):
    """Return k of FedMD-NFDP

    Args:
        epsilon (float): epsilon
        n (int): training set size
        replacement (bool, optional): sampling w/o replacement. Defaults to True.

    Returns:
        int: k
    """
    if replacement:
        return int(epsilon / math.log((n + 1) / n))
    else:
        return int((n + 1) * (1 - 1 / math.exp(epsilon)))
