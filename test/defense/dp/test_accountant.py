import pytest


def test_rdp_tight_upperbound():
    from aijack.defense import GeneralMomentAccountant

    sigma = 1.5
    sampling_rate = 0.04
    num_steps = int(90 / 0.04)

    accountant = GeneralMomentAccountant(
        noise_type="Gaussian",
        search="greedy",
        orders=list(range(2, 73)),
        max_iterations=100,
        bound_type="rdp_tight_upperbound",
        backend="cpp",
    )

    accountant.reset_step_info()
    accountant.add_step_info({"sigma": sigma}, sampling_rate, num_steps)
    assert accountant.get_epsilon(delta=1e-5) == pytest.approx(7.32911117143, 1e-2)

    target_delta = 1e-5
    sampling_rate = 0.04
    target_epsilon = 8
    num_iterations = 90
    assert (
        accountant.get_noise_multiplier(
            "sigma",
            target_epsilon,
            target_delta,
            sampling_rate,
            num_iterations,
            noise_multiplier_min=0,
            noise_multiplier_max=10,
            noise_multiplier_precision=0.0001,
        )
        == pytest.approx(1.425307617, 0.01)
    )


def test_rdp_upperbound_closedformula():
    from aijack.defense import GeneralMomentAccountant

    sigma = 1.5
    sampling_rate = 0.04
    num_steps = int(90 / 0.04)

    accountant = GeneralMomentAccountant(
        noise_type="Gaussian",
        search="ternary",
        precision=0.001,
        order_max=0,
        order_min=72,
        max_iterations=100,
        bound_type="rdp_upperbound_closedformula",
        backend="python",
    )

    accountant.reset_step_info()
    accountant.add_step_info({"sigma": sigma}, sampling_rate, num_steps)
    assert accountant.get_epsilon(delta=1e-5) == pytest.approx(7.32911117143, 1e-2)

    target_delta = 1e-5
    sampling_rate = 0.04
    target_epsilon = 8
    num_iterations = 90
    assert (
        accountant.get_noise_multiplier(
            "sigma",
            target_epsilon,
            target_delta,
            sampling_rate,
            num_iterations,
            noise_multiplier_min=0,
            noise_multiplier_max=10,
            noise_multiplier_precision=0.0001,
        )
        == pytest.approx(1.425307617, 0.01)
    )
