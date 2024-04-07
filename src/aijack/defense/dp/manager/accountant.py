import numpy as np

from aijack_cpp_core import (
    _greedy_search,
    _greedy_search_frac,
    _ternary_search,
    _ternary_search_int,
    calc_tightupperbound_lowerbound_of_rdp_with_theorem6and8_of_zhu_2019,
    calc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism,
    eps_gaussian,
    eps_laplace,
)

from .rdp import (
    calc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism as calc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism_py,
)


class BaseMomentAccountant:
    """
    Base class for computing the privacy budget using the Moments Accountant technique.
    """

    def __init__(
        self,
        search="ternary",
        order_min=2,
        order_max=64,
        precision=0.5,
        orders=[],
        max_iterations=10000,
    ):
        """
        Initialize the BaseMomentAccountant.

        Args:
            search (str, optional): The search strategy. Defaults to "ternary".
            order_min (int, optional): Minimum order. Defaults to 2.
            order_max (int, optional): Maximum order. Defaults to 64.
            precision (float, optional): Precision of the search. Defaults to 0.5.
            orders (list, optional): List of orders. Defaults to [].
            max_iterations (int, optional): Maximum number of iterations. Defaults to 10000.
        """
        self.order_min = order_min
        self.order_max = order_max
        self.orders = orders
        self.precision = precision
        self.max_iterations = max_iterations

        self.steps_info = []
        self.calc_bound_of_rdp = None

        if search == "ternary":
            self.search = _ternary_search
        elif search == "ternary_int":
            self.search = _ternary_search_int
        elif search == "greedy":
            self.search = _greedy_search
        elif search == "greedy_frac":
            self.search = _greedy_search_frac

        self._cache = {}

    def calc_upperbound_of_rdp_onestep(self, alpha, noise_params, sampling_rate):
        """
        Calculate the upper bound of Renyi Differential Privacy (RDP) for one step.

        Args:
            alpha (float): Privacy parameter alpha.
            noise_params (dict): Parameters of the noise distribution.
            sampling_rate (float): Sampling rate.

        Returns:
            float: Upper bound of RDP for one step.
        """
        key = hash(
            f"{alpha}_{list(noise_params.keys())[0]}_{list(noise_params.values())[0]}_{sampling_rate}"
        )
        if key not in self._cache:
            if sampling_rate == 0:
                result = 0
            elif sampling_rate == 1:
                result = self.eps_func(alpha, noise_params)
            else:
                result = self.calc_bound_of_rdp(
                    alpha, noise_params, sampling_rate, self.eps_func
                )
            self._cache[key] = result
            return result
        else:
            return self._cache[key]

    def _calc_upperbound_of_rdp(self, lam, steps_info):
        """
        Calculate the upper bound of RDP.

        Args:
            lam (float): Parameter lambda.
            steps_info (list): Information about steps.

        Returns:
            float: Upper bound of RDP.
        """
        rdp = 0.0
        for noise_params, sampling_rate, num_steps in steps_info:
            rdp += num_steps * self.calc_upperbound_of_rdp_onestep(
                lam, noise_params, sampling_rate
            )
        return rdp

    def reset_step_info(self):
        """Reset step information."""
        self.steps_info = []

    def add_step_info(self, noise_params, sampling_rate, num_steps):
        """
        Add step information.

        Args:
            noise_params (dict): Parameters of the noise distribution.
            sampling_rate (float): Sampling rate.
            num_steps (int): Number of steps.
        """
        self.steps_info.append((noise_params, sampling_rate, num_steps))

    def step(self, noise_params, sampling_rate, num_steps):
        """
        Decorator to add step information to a function.

        Args:
            noise_params (dict): Parameters of the noise distribution.
            sampling_rate (float): Sampling rate.
            num_steps (int): Number of steps.

        Returns:
            function: Decorated function.
        """

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
        """Get noise multiplier.

        Args:
            noise_multiplier_key (str): Key for noise multiplier.
            target_epsilon (float): Target epsilon.
            target_delta (float): Target delta.
            sampling_rate (float): Sampling rate.
            num_iterations (int): Number of iterations.
            noise_multiplier_min (float, optional): Minimum noise multiplier. Defaults to 0.
            noise_multiplier_max (float, optional): Maximum noise multiplier. Defaults to 10.
            noise_multiplier_precision (float, optional): Precision of noise multiplier. Defaults to 0.01.

        Returns:
            float: Noise multiplier.
        """
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
        """
        Get delta.

        Args:
            epsilon (float): Epsilon value.

        Returns:
            float: Delta value.
        """
        optimal_lam = self.search(
            lambda order: (order - 1)
            * (self._calc_upperbound_of_rdp(order - 1, self.steps_info) - epsilon),
            self.order_min,
            self.order_max,
            self.orders,
            self.precision,
            self.max_iterations,
        )

        min_delta = np.exp(
            (optimal_lam - 1)
            * (self._calc_upperbound_of_rdp(optimal_lam - 1, self.steps_info) - epsilon)
        )

        return min_delta

    def get_epsilon(self, delta):
        """
        Get epsilon.

        Args:
            delta (float): Delta value.

        Returns:
            float: Epsilon value.
        """
        # log_inv_delta = math.log(1 / delta)

        def estimate_eps(order):
            return (
                self._calc_upperbound_of_rdp(order, self.steps_info)
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
    """
    Generalized class for computing the privacy budget using the Moments Accountant technique.
    """

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
        """
        Initialize the GeneralMomentAccountant.

        Args:
            name (str, optional): Name of the accountant. Defaults to "SGM".
            search (str, optional): The search strategy. Defaults to "ternary".
            order_min (int, optional): Minimum order. Defaults to 2.
            order_max (int, optional): Maximum order. Defaults to 64.
            precision (float, optional): Precision of the search. Defaults to 0.5.
            orders (list, optional): List of orders. Defaults to [].
            noise_type (str, optional): Type of noise. Defaults to "Gaussian".
            bound_type (str, optional): Type of bound. Defaults to "rdp_upperbound_closedformula".
            max_iterations (int, optional): Maximum number of iterations. Defaults to 10000.
            backend (str, optional): Backend for calculation. Defaults to "cpp".
        """
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
        """
        Set the noise type.

        Args:
            noise_type (str): Type of noise.
        """
        if noise_type == "Gaussian":
            self.eps_func = eps_gaussian
        elif noise_type == "Laplace":
            self.eps_func = eps_laplace

    def _set_upperbound_func(self, backend, bound_type):
        """
        Set the upper bound function.

        Args:
            backend (str): Backend for calculation.
            bound_type (str): Type of bound.
        """
        if backend == "cpp" and bound_type == "rdp_upperbound_closedformula":
            self.calc_bound_of_rdp = (
                calc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism
            )
        elif backend == "python" and bound_type == "rdp_upperbound_closedformula":
            self.calc_bound_of_rdp = (
                calc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism_py
            )
        elif backend == "cpp" and bound_type == "rdp_tight_upperbound":
            self.calc_bound_of_rdp = (
                calc_tightupperbound_lowerbound_of_rdp_with_theorem6and8_of_zhu_2019
            )
