import math
from math import log, sqrt

import numpy as np
import torch
from statsmodels.stats.proportion import proportion_confint

log1c25 = log(1.25)


def clopper_pearson_interval(num_success, num_total, alpha):
    """
    Calculate the Clopper-Pearson confidence interval.

    Args:
        num_success (int): Number of successes.
        num_total (int): Total number of trials.
        alpha (float): Significance level.

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    return proportion_confint(num_success, num_total, alpha=2 * alpha, method="beta")


def gaus_delta_term(delta):
    """
    Calculate the Gaussian delta term.

    Args:
        delta (float): Delta value.

    Returns:
        float: Gaussian delta term.
    """
    return sqrt(2 * (log1c25 - log(delta)))


def get_maximum_L_laplace(lower_bound, upper_bound, L, dp_eps):
    """
    Calculate the maximum L value for Laplace mechanism.

    Args:
        lower_bound (float): Lower bound of the confidence interval.
        upper_bound (float): Upper bound of the confidence interval.
        L (float): Sensitivity parameter.
        dp_eps (float): Epsilon value for differential privacy.

    Returns:
        float: Maximum L value.
    """
    if lower_bound <= upper_bound:
        return 0.0
    return L * log(lower_bound / upper_bound) / (2 * dp_eps)


def get_maximum_L_gaussian(
    p_max_lb,
    p_sec_ub,
    attack_size,
    dp_epsilon,
    dp_delta,
    delta_range=None,
    eps_min=0.0,
    eps_max=1.0,
    tolerance=0.001,
):
    """
    Calculate the maximum L value for Gaussian mechanism.

    Args:
        p_max_lb (float): Lower bound of the maximum probability.
        p_sec_ub (float): Upper bound of the second maximum probability.
        attack_size (float): Size of the attack.
        dp_epsilon (float): Epsilon value for differential privacy.
        dp_delta (float): Delta value for differential privacy.
        delta_range (list, optional): Range of delta values. Defaults to None.
        eps_min (float, optional): Minimum epsilon value. Defaults to 0.0.
        eps_max (float, optional): Maximum epsilon value. Defaults to 1.0.
        tolerance (float, optional): Tolerance for epsilon search. Defaults to 0.001.

    Returns:
        float: Maximum L value.
    """
    # Based on the original implementation:
    # https://github.com/columbia/pixeldp/blob/master/models/utils/robustness.py
    if p_max_lb <= p_sec_ub:
        return 0.0

    if delta_range is None:
        delta_range = list(np.arange(0.001, 0.3, 0.001))

    max_r = 0.0
    for delta in delta_range:
        eps = (eps_min + eps_max) / 2
        while eps_min < eps and eps_max >= eps:
            L_temp = (
                attack_size
                * (eps / dp_epsilon)
                * (gaus_delta_term(dp_delta) / gaus_delta_term(delta))
            )
            if p_max_lb >= math.e ** (2 * eps) * p_sec_ub + (1 + math.e**eps) * delta:
                if L_temp > max_r:
                    max_r = L_temp
                    eps_min = eps
                    eps = (eps_min + eps_max) / 2.0
            else:
                # eps is too big for delta
                eps_max = eps
                eps = (eps_min + eps_max) / 2.0

            if eps_max - eps_min < tolerance:
                break

    return max_r


def get_certified_robustness_size_argmax(counts, eta, L, eps, delta, mode="gaussian"):
    """
    Calculate the maximum certified robustness size.

    Args:
        counts (torch.Tensor): Count of predictions.
        eta (float): Eta value.
        L (float): Sensitivity parameter.
        eps (float): Epsilon value.
        delta (float): Delta value.
        mode (str, optional): Mode of calculation. Defaults to "gaussian".

    Returns:
        float: Maximum certified robustness size.
    """
    total_counts = torch.sum(counts)
    sorted_counts, _ = torch.sort(counts)
    lb = clopper_pearson_interval(sorted_counts[-1], total_counts, eta)[0]
    ub = clopper_pearson_interval(sorted_counts[-2], total_counts, eta)[1]

    if mode == "laplace":
        return get_maximum_L_laplace(lb, ub, L, eps)
    elif mode == "gaussian":
        return get_maximum_L_gaussian(lb, ub, L, eps, delta)
    else:
        raise ValueError(f"{mode} is not supported")


class PixelDP(torch.nn.Module):
    """Implementation of Lecuyer, Mathias, et al. 'Certified robustness to
    adversarial examples with differential privacy.' 2019 IEEE symposium
    on security and privacy (SP). IEEE, 2019."""

    def __init__(
        self,
        model,
        num_classes,
        L,
        eps,
        delta,
        n_population_mc=1000,
        batch_size_mc=32,
        eta=0.05,
        mode="laplace",
        sensitivity=1,
    ):
        """
        Initialize the PixelDP module.

        Args:
            model (torch.nn.Module): The model to be used.
            num_classes (int): Number of classes.
            L (float): Sensitivity parameter.
            eps (float): Epsilon value.
            delta (float): Delta value.
            n_population_mc (int, optional): Number of samples for Monte Carlo. Defaults to 1000.
            batch_size_mc (int, optional): Batch size for Monte Carlo. Defaults to 32.
            eta (float, optional): Eta value. Defaults to 0.05.
            mode (str, optional): Mode of operation. Defaults to "laplace".
            sensitivity (float, optional): Sensitivity value. Defaults to 1.
        """
        super(PixelDP, self).__init__()

        self.model = model
        self.num_classes = num_classes
        self.L = L
        self.eps = eps
        self.delta = delta
        self.n_population_mc = n_population_mc
        self.batch_size_mc = batch_size_mc
        self.eta = eta
        self.mode = mode

        if self.mode == "laplace":
            self.sigma = sensitivity * L / eps
            self.dist = torch.distributions.laplace.Laplace(0, self.sigma)
        elif self.mode == "gaussian":
            self.sigma = gaus_delta_term(delta) * sensitivity * L / eps
        else:
            raise ValueError(f"{mode} is not supported")

    def sample_noise(self, x):
        """
        Sample noise for the given input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Sampled noise.
        """
        if self.mode == "laplace":
            return self.dist.sample(x.shape).to(x.device)
        else:
            return torch.randn_like(x, device=x.device) * self.sigma

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_eval(x)

    def forward_train(self, x):
        return self.model(x + self.sample_noise(x))

    def forward_eval(self, x):
        remaining_num = self.n_population_mc
        input_dim = len(x.shape) - 1
        with torch.no_grad():
            preds = torch.zeros(self.num_classes, device=x.device)
            counts = torch.zeros(self.num_classes, device=x.device)
            while remaining_num > 0:
                batch_size = min(self.batch_size_mc, remaining_num)
                remaining_num -= batch_size
                inputs = x.repeat(tuple([batch_size] + [1] * input_dim))
                noise = self.sample_noise(inputs)
                tmp_pred = self.model(inputs + noise)
                tmp_pred_argmax = tmp_pred.argmax(1)

                preds += tmp_pred.sum(0)
                for i in range(len(tmp_pred_argmax)):
                    counts[tmp_pred_argmax[i]] += 1

        return preds, counts

    def certify(self, counts):
        """
        Certify the robustness of the model.

        Args:
            counts (torch.Tensor): Count of predictions.

        Returns:
            float: Certified robustness size.
        """
        return get_certified_robustness_size_argmax(
            counts, self.eta, self.L, self.eps, self.delta, self.mode
        )
