import math

import numpy as np
from scipy import special
from scipy.special import logsumexp

from .utils import _log_add, _log_erfc, _log_sub


def eps_gaussian(alpha, params):
    if alpha == math.inf:
        return min(
            4 * (np.exp(eps_gaussian(2, params) - 1)),
            2 * np.exp(eps_gaussian(2, params)),
        )
    return alpha / (2 * (params["sigma"] ** 2))


def eps_laplace(alpha, params):
    if alpha <= 1:
        return 1 / params["b"] + np.exp(-1 / params["b"]) - 1
    elif alpha == math.inf:
        return 1 / params["b"]
    return (1 / (alpha - 1)) * logsumexp(
        [
            np.log(alpha / (2 * alpha - 1)) + (alpha - 1) / params["b"],
            np.log((alpha - 1) / (2 * alpha - 1)) + (-alpha) / params["b"],
        ],
        b=[1, 1],
    )


def eps_randresp(alpha, params):
    if params["p"] == 1 or params["p"] == 0:
        return math.inf
    if alpha <= 1:
        return (2 * params["p"] - 1) * np.log(params["p"] / (1 - params["p"]))
    elif alpha == math.inf:
        return np.abs(np.log((1.0 * params["p"] / (1 - params["p"]))))
    return (1 / (alpha - 1)) * logsumexp(
        [
            alpha * np.log(params["p"]) + (1 - alpha) * np.log(1 - params["p"]),
            alpha * np.log(1 - params["p"]) + (1 - alpha) * params["p"],
        ],
        b=[1, 1],
    )


def culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism(
    alpha, params, sampling_rate, _eps
):
    """Compute log(A_alpha) for any positive finite alpha."""
    if float(alpha).is_integer():
        return culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism_int(
            int(alpha), params, sampling_rate
        )
    else:
        return culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism_float(
            alpha, params, sampling_rate
        )


def culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism_int(
    alpha, params, sampling_rate
):
    """Renyi Differential Privacy of the Sampled Gaussian Mechanism
    3.3 Numerically Stable Computatio
    """
    log_a = -np.inf

    for i in range(alpha + 1):
        log_coef_i = (
            np.log(special.binom(alpha, i))
            + i * np.log(sampling_rate)
            + (alpha - i) * np.log(1 - sampling_rate)
        )

        s = log_coef_i + (i * i - i) / (2 * (params["sigma"] ** 2))
        log_a = _log_add(log_a, s)

    return float(log_a) / (alpha - 1)


def culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism_float(
    alpha, params, sampling_rate
):
    """Compute log(A_alpha) for fractional alpha. 0 < q < 1."""
    # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
    # initialized to 0 in the log space:
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0

    z0 = params["sigma"] ** 2 * np.log(1 / sampling_rate - 1) + 0.5

    while True:  # do ... until loop
        coef = special.binom(alpha, i)
        log_coef = np.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * np.log(sampling_rate) + j * np.log(1 - sampling_rate)
        log_t1 = log_coef + j * np.log(sampling_rate) + i * np.log(1 - sampling_rate)

        log_e0 = np.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * params["sigma"]))
        log_e1 = np.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * params["sigma"]))

        log_s0 = log_t0 + (i * i - i) / (2 * (params["sigma"] ** 2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (params["sigma"] ** 2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1) / (alpha - 1)


def culc_upperbound_of_rdp_with_theorem27_of_wang_2019(
    alpha, params, sampling_rate, _eps
):
    def B(el):
        res = 0
        for i in range(el + 1):
            res += (
                ((-1) ** (i)) * special.binom(el, i) * np.exp((i - 1) * _eps(i, params))
            )
        return abs(res)

    """
    def logAbsB(el):
        ts = []
        ss = []
        for i in range(el + 1):
            ts.append(np.log(special.binom(el, i)) + (i - 1) * _eps(i, params))
            ss.append((-1) ** (i + el % 2))

        return logsumexp(ts, b=ss)
    """

    terms = []
    signs = []

    terms.append(np.log(1))
    signs.append(1)

    second = (
        2 * np.log(sampling_rate)
        + np.log(special.binom(alpha, 2))
        + min(
            np.log(4) + logsumexp([_eps(2, params), np.log(1)], b=[1, -1]),
            _eps(2, params)
            + min(
                np.log(2),
                2 * logsumexp([_eps(math.inf, params), np.log(1)], b=[1, -1]),
            ),
        )
    )
    terms.append(second)
    signs.append(1)

    for j in range(3, alpha + 1):
        terms.append(
            np.log(4)
            + j * np.log(sampling_rate)
            + np.log(special.binom(alpha, j))
            + (1 / 2)
            * (np.log(B(2 * math.floor(j / 2))) + np.log(B(2 * math.ceil(j / 2))))
        )
        signs.append(1)

    return (1 / (alpha - 1)) * logsumexp(terms, b=signs)


def culc_general_upperbound_of_rdp_with_theorem5_of_zhu_2019(
    alpha, params, sampling_rate, _eps
):

    terms = []
    signs = []

    first = ((alpha - 1) * np.log(1 - sampling_rate)) + np.log(
        alpha * sampling_rate - sampling_rate + 1
    )
    terms.append(first)
    signs.append(1)

    if alpha >= 2:
        second = (
            np.log(special.binom(alpha, 2))
            + (2 * np.log(sampling_rate))
            + ((alpha - 2) * np.log(1 - sampling_rate))
            + _eps(2, params)
        )
        terms.append(second)
        signs.append(1)

    if alpha >= 3:
        for el in range(3, alpha + 1):
            third = np.log(3) + (
                np.log(special.binom(alpha, el))
                + (alpha - el) * np.log(1 - sampling_rate)
                + (el * np.log(sampling_rate))
                + ((el - 1) * _eps(el, params))
            )
            terms.append(third)
            signs.append(1)

    return (1 / (alpha - 1)) * logsumexp(terms, b=signs)


def culc_tightupperbound_lowerbound_of_rdp_with_theorem6and8_of_zhu_2019(
    alpha, params, sampling_rate, _eps
):
    terms = []
    signs = []

    first = ((alpha - 1) * np.log(1 - sampling_rate)) + np.log(
        alpha * sampling_rate - sampling_rate + 1
    )
    terms.append(first)
    signs.append(1)

    if alpha >= 2:
        for el in range(2, alpha + 1):
            second = (
                np.log(special.binom(alpha, el))
                + (el * np.log(sampling_rate))
                + ((alpha - el) * np.log(1 - sampling_rate))
                + ((el - 1) * _eps(el, params))
            )
            terms.append(second)
            signs.append(1)

    return (1 / (alpha - 1)) * logsumexp(terms, b=signs)


def culc_tightupperbound_lowerbound_of_rdp_with_theorem6and8_of_zhu_2019_with_tau_estimation(
    alpha, params, sampling_rate, _eps, tau=10
):

    terms = []
    signs = []

    eps_alpha_minus_tau = _eps(alpha - tau, params)

    first = alpha * np.log(1 - sampling_rate) + logsumexp(
        [np.log(1), -eps_alpha_minus_tau], b=[1, -1]
    )
    terms.append(first)
    signs.append(1)

    second = -eps_alpha_minus_tau + alpha * logsumexp(
        [
            np.log(1),
            np.log(sampling_rate),
            np.log(sampling_rate) + eps_alpha_minus_tau,
        ],
        b=[1, -1, 1],
    )
    terms.append(second)
    signs.append(1)

    for el in range(2, tau):
        third = (
            np.log(special.binom(alpha, el))
            + (alpha - el) * np.log(1 - sampling_rate)
            + el * np.log(sampling_rate)
            + logsumexp(
                [(el - 1) * eps_alpha_minus_tau, (el - 1) * _eps(el, params)], b=[1, -1]
            )
        )
        terms.append(third)
        signs.append(-1)

    for el in range(alpha - tau + 1, alpha + 1):
        fourth = (
            np.log(special.binom(alpha, el))
            + (alpha - el) * np.log(1 - sampling_rate)
            + el * np.log(sampling_rate)
            + logsumexp(
                [(el - 1) * _eps(el, params), (el - 1) * eps_alpha_minus_tau],
                b=[1, -1],
            )
        )
        terms.append(fourth)
        signs.append(1)

    return (1 / (alpha - 1)) * logsumexp(terms, b=signs)
