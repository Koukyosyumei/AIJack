import torch
import math
from math import sqrt, log
from statsmodels.stats.proportion import proportion_confint
import numpy as np


def clopper_pearson_interval(self, num_success, num_total, alpha):
    return proportion_confint(num_success, num_total,
                              alpha=2*alpha, method="beta")


def hoeffding_interval(self, e_ax, num_total, alpha, k):
    bound = sqrt(log(2*k/alpha) / (2*num_total))
    return e_ax - bound, e_ax + bound


log1c25 = log(1.25)


def gaus_delta_term(delta):
    return sqrt(2 * (log1c25 - log(delta)))


def get_maximum_L_laplace(lower_bound, upper_bound, L, dp_eps):
    if lower_bound <= upper_bound:
        return 0.0
    return L * log(lower_bound / upper_bound) / (2 * dp_eps)


def get_maximum_L_gaussian(p_max_lb, p_sec_ub, attack_size,
                           dp_epsilon, dp_delta,
                           delta_range=list(np.arange(0.001, 0.3, 0.001)),
                           eps_min=0.0, eps_max=1.0, tolerance=0.001):
    if p_max_lb <= p_sec_ub:
        return 0.0

    max_r = 0.0
    for delta in delta_range:
        eps = (eps_min + eps_max) / 2
        while eps_min < eps and eps_max >= eps:
            L_temp = attack_size *  \
                (eps / dp_epsilon) *  \
                (gaus_delta_term(
                    dp_delta) / gaus_delta_term(delta))
            if p_max_lb >= math.e ** (2 * eps) *\
                    p_sec_ub + (1 +
                                math.e ** eps) * delta:
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


def get_certified_robustness_size_argmax(counts, eta, L, eps,
                                         delta, mode="gaussian"):
    total_counts = torch.sum(counts)
    sorted_counts, _ = torch.sort(counts)
    lb = clopper_pearson_interval(sorted_counts[-1], total_counts, eta)
    ub = clopper_pearson_interval(sorted_counts[-2], total_counts, eta)

    if mode == "laplace":
        return get_maximum_L_laplace(lb, ub, L, eps)
    elif mode == "gaussian":
        return get_maximum_L_gaussian(lb, ub, L, eps, delta)
    else:
        raise ValueError(f"{mode} is not supported")


class PixelDP(torch.nn.Module):
    def __init__(self, model, num_classes,
                 L, eps, delta,
                 n_population_mc, batch_size_mc, eta,
                 mode="laplace",
                 sensitivity=1):
        self.model = model
        self.num_classes = num_classes
        self.eps = eps
        self.delta = delta
        self.n_population_mc = n_population_mc
        self.batch_size_mc = batch_size_mc
        self.eta = eta
        self.mode = mode

        if self.mode == "laplace":
            self.sigma = sensitivity * L / eps
        elif self.mode == "gaussian":
            self.sigma = gaus_delta_term(delta) * sensitivity * L / eps
        else:
            raise ValueError(f"{mode} is not supported")

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_eval(x)

    def forward_train(self, x):
        return self.model(x + torch.randn_like(x, self.device) * self.sigma)

    def forward_eval(self, x):
        remaining_num = self.n_population_mc
        with torch.no_grad():
            preds = torch.zeros(self.num_classes)
            counts = torch.zeros(self.num_classes)
            while remaining_num > 0:
                batch_size = min(self.batch_size_mc, remaining_num)
                remaining_num -= batch_size
                inputs = x.repeat((batch_size, 1, 1, 1))
                noise = torch.randn_like(
                    inputs, device=self.device) * self.sigma
                tmp_pred = self.model(inputs + noise)
                tmp_pred_argmax = tmp_pred.argmax(1)

                preds += tmp_pred
                for i in range(len(tmp_pred_argmax)):
                    counts[tmp_pred_argmax[i]] += 1

        return preds, counts

    def certify(self, counts):
        return get_certified_robustness_size_argmax(counts,
                                                    self.eta, self.L,
                                                    self.eps, self.delta,
                                                    self.mode)
