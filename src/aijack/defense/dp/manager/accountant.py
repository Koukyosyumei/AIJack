import numpy as np

from aijack_dp_core import (
    _greedy_search,
    _greedy_search_frac,
    _ternary_search,
    _ternary_search_int,
    culc_tightupperbound_lowerbound_of_rdp_with_theorem6and8_of_zhu_2019,
    culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism,
    eps_gaussian,
    eps_laplace,
)

from .rdp import (
    culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism as culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism_py,
)


class BaseMomentAccountant:
    def __init__(
        self,
        search="ternary",
        order_min=2,
        order_max=64,
        precision=0.5,
        orders=[],
        max_iterations=10000,
    ):
        self.order_min = order_min
        self.order_max = order_max
        self.orders = orders
        self.precision = precision
        self.max_iterations = max_iterations

        self.steps_info = []
        self._culc_bound_of_rdp = None

        if search == "ternary":
            self.search = _ternary_search
        elif search == "ternary_int":
            self.search = _ternary_search_int
        elif search == "greedy":
            self.search = _greedy_search
        elif search == "greedy_frac":
            self.search = _greedy_search_frac

        self._cache = {}

    def _culc_upperbound_of_rdp_onestep(self, alpha, noise_params, sampling_rate):
        key = hash(
            f"{alpha}_{list(noise_params.keys())[0]}_{list(noise_params.values())[0]}_{sampling_rate}"
        )
        if key not in self._cache:
            if sampling_rate == 0:
                result = 0
            elif sampling_rate == 1:
                result = self.eps_func(alpha, noise_params)
            else:
                result = self._culc_bound_of_rdp(
                    alpha, noise_params, sampling_rate, self.eps_func
                )
            self._cache[key] = result
            return result
        else:
            return self._cache[key]

    def _culc_upperbound_of_rdp(self, lam, steps_info):
        rdp = 0.0
        for noise_params, sampling_rate, num_steps in steps_info:
            rdp += num_steps * self._culc_upperbound_of_rdp_onestep(
                lam, noise_params, sampling_rate
            )
        return rdp

    def reset_step_info(self):
        self.steps_info = []

    def add_step_info(self, noise_params, sampling_rate, num_steps):
        self.steps_info.append((noise_params, sampling_rate, num_steps))

    def step(self, noise_params, sampling_rate, num_steps):
        def _step(f):
            def _wrapper(*args, **keywords):
                result = f(*args, **keywords)
                self.add_step_info(self, noise_params, sampling_rate, num_steps)
                return result

            return _wrapper

        return _step

    def get_noise_multiplier(
        self,
        noise_multiplier_key,
        target_epsilon,
        target_delta,
        sampling_rate,
        num_iterations,
        noise_multiplier_min=0,
        noise_multiplier_max=10,
        noise_multiplier_precision=0.01,
    ):
        eps = float("inf")
        while eps > target_epsilon:
            noise_multiplier_max = 2 * noise_multiplier_max
            self.steps_info = [
                (
                    {noise_multiplier_key: noise_multiplier_max},
                    sampling_rate,
                    int(num_iterations / sampling_rate),
                )
            ]
            eps = self.get_epsilon(target_delta)

        while (
            noise_multiplier_max - noise_multiplier_min
        ) > noise_multiplier_precision:
            noise_multiplier = (noise_multiplier_max + noise_multiplier_min) / 2
            self.steps_info = [
                (
                    {noise_multiplier_key: noise_multiplier},
                    sampling_rate,
                    int(num_iterations / sampling_rate),
                )
            ]
            eps = self.get_epsilon(target_delta)

            if eps < target_epsilon:
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
            self.max_iterations,
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

        optimal_lam = self.search(
            estimate_eps,
            self.order_min,
            self.order_max,
            self.orders,
            self.precision,
            self.max_iterations,
        )

        min_epsilon = estimate_eps(optimal_lam)

        return min_epsilon


class GeneralMomentAccountant(BaseMomentAccountant):
    def __init__(
        self,
        name="SGM",
        search="ternary",
        order_min=2,
        order_max=64,
        precision=0.5,
        orders=[],
        noise_type="Gaussian",
        bound_type="rdp_upperbound_closedformula",
        max_iterations=10000,
        backend="cpp",
    ):
        super().__init__(
            search=search,
            order_min=order_min,
            order_max=order_max,
            precision=precision,
            orders=orders,
            max_iterations=max_iterations,
        )
        self.name = name
        self._set_noise_type(noise_type)
        self._set_upperbound_func(backend, bound_type)

    def _set_noise_type(self, noise_type):
        if noise_type == "Gaussian":
            self.eps_func = eps_gaussian
        elif noise_type == "Laplace":
            self.eps_func = eps_laplace

    def _set_upperbound_func(self, backend, bound_type):
        if backend == "cpp" and bound_type == "rdp_upperbound_closedformula":
            self._culc_bound_of_rdp = (
                culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism
            )
        elif backend == "python" and bound_type == "rdp_upperbound_closedformula":
            self._culc_bound_of_rdp = (
                culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism_py
            )
        elif backend == "cpp" and bound_type == "rdp_tight_upperbound":
            self._culc_bound_of_rdp = (
                culc_tightupperbound_lowerbound_of_rdp_with_theorem6and8_of_zhu_2019
            )
