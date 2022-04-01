import pytest


def test_FedMD_NFDP():
    from aijack.collaborative.fedmd import (
        get_epsilon_of_fedmd_nfdp,
        get_k_of_fedmd_nfdp,
        get_sigma_of_fedmd_nfdp,
    )

    assert get_k_of_fedmd_nfdp(
        get_epsilon_of_fedmd_nfdp(300, 100, replacement=True), 300, replacement=True
    ) == pytest.approx(100, 0)

    # assert get_k_of_fedmd_nfdp(
    #    get_epsilon_of_fedmd_nfdp(300, 100, replacement=False), 300, replacement=False
    # ) == pytest.approx(100, 0)

    assert get_k_of_fedmd_nfdp(
        get_epsilon_of_fedmd_nfdp(300, 300, replacement=True), 300, replacement=True
    ) == pytest.approx(300, 0)

    assert get_k_of_fedmd_nfdp(
        get_epsilon_of_fedmd_nfdp(300, 300, replacement=False), 300, replacement=False
    ) == pytest.approx(300, 0)

    assert get_sigma_of_fedmd_nfdp(300, 120, replacement=True) == pytest.approx(
        0.3301, 1e-4
    )
