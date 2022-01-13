import numpy as np

from aijack_dp_core import (
    culc_tightupperbound_lowerbound_of_rdp_with_theorem6and8_of_zhu_2019,
    culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism,
    eps_gaussian,
    eps_laplace,
)

from .search import (
    _bisection_search,
    _greedy_search,
    _ternary_search,
    _ternary_search_int,
)


class BaseMomentAccountant:
    def __init__(
        self, search="ternary", order_min=2, order_max=64, precision=0.5, orders=None
    ):
        self.order_min = order_min
        self.order_max = order_max
        self.orders = orders
        self.precision = precision

        self.steps_info = []
        self._culc_bound_of_rdp = None

        if search == "ternary":
            self.search = _ternary_search
        elif search == "ternary_int":
            self.search = _ternary_search_int
        elif search == "bisection":
            self.search = _bisection_search
        elif search == "greedy":
            self.search = _greedy_search

    def _culc_upperbound_of_rdp_onestep(self, alpha, noise_params, sampling_rate):
        if sampling_rate == 0:
            return 0
        elif sampling_rate == 1:
            return self.eps_func(alpha, noise_params)
        else:
            return self._culc_bound_of_rdp(
                alpha, noise_params, sampling_rate, self.eps_func
            )

    def _culc_upperbound_of_rdp(self, lam, steps_info):
        rdp = 0.0
        for (noise_params, sampling_rate, num_steps) in steps_info:
            rdp += num_steps * self._culc_upperbound_of_rdp_onestep(
                lam, noise_params, sampling_rate
            )
        return rdp

    def reset_step_info(self):
        self.steps_info = []

    def add_step_info(self, noise_params, sampling_rate, num_steps):
        self.steps_info.append((noise_params, sampling_rate, num_steps))

    def get_noise_multiplier(
        self,
        noise_multiplier_key,
        noise_multiplier_min,
        noise_multiplier_max,
        eps_max,
        delta,
        sampling_rate,
        num_iterations,
        noise_multiplier_precision=0.01,
    ):
        eps = float("inf")
        while eps > eps_max:
            noise_multiplier_max = 2 * noise_multiplier_max
            self.steps = [
                (
                    {noise_multiplier_key: noise_multiplier_max},
                    sampling_rate,
                    int(num_iterations / sampling_rate),
                )
            ]
            eps = self.get_epsilon(delta)

        while (
            noise_multiplier_max - noise_multiplier_min
        ) > noise_multiplier_precision:
            noise_multiplier = (noise_multiplier_max + noise_multiplier_min) / 2
            self.steps = [
                (
                    {noise_multiplier_key: noise_multiplier},
                    sampling_rate,
                    int(num_iterations / sampling_rate),
                )
            ]
            eps = self.get_epsilon(delta)

            if eps < eps_max:
                noise_multiplier_max = noise_multiplier
            else:
                noise_multiplier_min = noise_multiplier

        return noise_multiplier

    def get_delta(self, epsilon):
        optimal_lam = self.search(
            lambda order: (order - 1)
            * (self._culc_upperbound_of_rdp(order - 1, self.steps_info) - epsilon),
            self.order_min,
            self.order_max,
            self.orders,
            self.precision,
        )

        min_delta = np.exp(
            (optimal_lam - 1)
            * (self._culc_upperbound_of_rdp(optimal_lam - 1, self.steps_info) - epsilon)
        )

        return min_delta

    def get_epsilon(self, delta):
        # log_inv_delta = math.log(1 / delta)

        def estimate_eps(order):
            return (
                self._culc_upperbound_of_rdp(order, self.steps_info)
                - (np.log(order) + np.log(delta)) / (order - 1)
                + np.log((order - 1) / order)
            )

        """

        def estimate_eps(order):
            return log_inv_delta / (order - 1) + self._culc_upperbound_of_rdp(
                order - 1, self.steps_info
            )
        """

        optimal_lam = self.search(
            estimate_eps,
            self.order_min,
            self.order_max,
            self.orders,
            self.precision,
        )

        min_epsilon = estimate_eps(optimal_lam)

        return min_epsilon


class GeneralMomentAccountant(BaseMomentAccountant):
    def __init__(
        self,
        name="",
        search="bisection",
        order_min=2,
        order_max=64,
        precision=0.5,
        orders=None,
        noise_type="Gaussian",
        bound_type="rdp_upperbound_closedformula",
    ):
        super().__init__(
            search=search,
            order_min=order_min,
            order_max=order_max,
            precision=precision,
            orders=orders,
        )
        self.name = name
        self._set_noise_type(noise_type)
        self._set_upperbound_func(bound_type)

    def _set_noise_type(self, noise_type):
        if noise_type == "Gaussian":
            self.eps_func = eps_gaussian
        elif noise_type == "Laplace":
            self.eps_func = eps_laplace

    def _set_upperbound_func(self, bound_type):
        if bound_type == "rdp_upperbound_closedformula":
            self._culc_bound_of_rdp = (
                culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism
            )
        elif bound_type == "rdp_tight_upperbound":
            self._culc_bound_of_rdp = (
                culc_tightupperbound_lowerbound_of_rdp_with_theorem6and8_of_zhu_2019
            )
