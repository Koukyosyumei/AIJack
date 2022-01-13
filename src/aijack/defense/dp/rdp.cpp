#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <cmath>
#include <iostream>
#include <vector>
#include "utils.cpp"

using namespace std;
namespace py = pybind11;

long double eps_gaussian(long double alpha, py::dict params)
{
    if (std::isinf(alpha))
    {
        return min(4 * std::exp(eps_gaussian(2, params) - 1),
                   2 * std::exp(eps_gaussian(2, params)));
    }
    return alpha / (2 * pow(params["sigma"].cast<long double>(), 2));
}

long double eps_laplace(long double alpha, py::dict params)
{
    long double b = params["b"].cast<long double>();
    if (alpha <= 1)
    {
        return 1 / b + std::exp(-1 / b) - 1;
    }
    else if (std::isinf(alpha))
    {
        return 1 / b;
    }
    else
    {
        std::vector<long double> val(2);
        std::vector<int> sgn(2);
        val[0] = std::log(alpha / (2 * alpha - 1)) + (alpha - 1) / b;
        sgn[0] = 1;
        val[1] = std::log((alpha - 1) / (2 * alpha - 1)) + (-1 * alpha) / b;
        sgn[1] = 1;
        return (1 / (alpha - 1)) * logsumexp(val, sgn);
    }
}

long double culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism_int(int alpha,
                                                                       py::dict params,
                                                                       long double sampling_rate)
{
    long double sigma = params["sigma"].cast<long double>();
    long double inv_double_sigma_square = 1 / (2 * (sigma * sigma));
    long double log_a = -1 * std::numeric_limits<long double>::infinity();

    for (int i = 0; i <= alpha; i++)
    {
        long double log_coef_i = std::log(binom(alpha, i)) +
                                 i * std::log(sampling_rate) +
                                 (alpha - i) * std::log(1 - sampling_rate);

        long double s = log_coef_i + (i * i - i) * inv_double_sigma_square;
        log_a = _log_add(log_a, s);
    }
    return log_a / (alpha - 1);
}

long double culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism_double(long double alpha,
                                                                          py::dict params,
                                                                          long double sampling_rate)
{
    long double sigma = params["sigma"].cast<long double>();
    long double inv_double_sigma_square = 1 / (2 * (sigma * sigma));
    long double log_sampling_rate = std::log(sampling_rate);
    long double log_1m_sampling_rate = std::log(1 - sampling_rate);
    long double log_half = std::log(0.5);

    long double log_a0 = -1 * std::numeric_limits<long double>::infinity();
    long double log_a1 = -1 * std::numeric_limits<long double>::infinity();
    long double i = 0;

    long double z0 = pow(sigma, 2) * std::log(1 / sampling_rate - 1) + 0.5;

    long double coef, log_coef, j, log_t0, log_t1, log_e0, log_e1, log_s0, log_s1;
    while (true)
    {
        coef = binom(alpha, i);
        log_coef = std::log(std::abs(coef));
        j = alpha - i;

        log_t0 = log_coef + i * log_sampling_rate + j * log_1m_sampling_rate;
        log_t1 = log_coef + j * log_sampling_rate + i * log_1m_sampling_rate;

        log_e0 = log_half + _log_erfc((i - z0) / (std::sqrt(2) * sigma));
        log_e1 = log_half + _log_erfc((z0 - j) / (std::sqrt(2) * sigma));

        log_s0 = log_t0 + (i * i - i) * inv_double_sigma_square + log_e0;
        log_s1 = log_t1 + (j * j - j) * inv_double_sigma_square + log_e1;

        if (coef > 0)
        {
            log_a0 = _log_add(log_a0, log_s0);
            log_a1 = _log_add(log_a1, log_s1);
        }
        else
        {
            log_a0 = _log_sub(log_a0, log_s0);
            log_a1 = _log_sub(log_a1, log_s1);
        }

        i += 1;
        if (max(log_s0, log_s1) < -30)
        {
            break;
        }
    }

    return _log_add(log_a0, log_a1) / (alpha - 1);
}

long double culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism(long double alpha,
                                                                   py::dict params,
                                                                   long double sampling_rate,
                                                                   const std::function<long double(long double, py::dict)> &_eps)
{
    if (fmod(alpha, 1) == 0.0)
    {
        return culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism_int(
            (int)alpha, params, sampling_rate);
    }
    else
    {
        return culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism_double(
            alpha, params, sampling_rate);
    }
}

long double culc_tightupperbound_lowerbound_of_rdp_with_theorem6and8_of_zhu_2019(int alpha,
                                                                                 py::dict params,
                                                                                 long double sampling_rate,
                                                                                 const std::function<long double(long double, py::dict)> &_eps)
{
    std::vector<long double> terms(alpha);
    std::vector<int> signs(alpha);

    long double first = ((alpha - 1) * std::log(1 - sampling_rate)) +
                        std::log(alpha * sampling_rate - sampling_rate + 1);
    terms[0] = first;
    signs[0] = 1;

    if (alpha >= 2)
    {
        long double second;
        for (int el = 2; el <= alpha; el++)
        {
            second = (std::log(binom(alpha, el)) +
                      (el * std::log(sampling_rate)) +
                      ((alpha - el) * std::log(1 - sampling_rate)) +
                      ((el - 1) * _eps(el, params)));
            terms[el - 1] = second;
            signs[el - 1] = 1;
        }
    }

    return (1 / ((long double)alpha - 1)) * logsumexp(terms, signs);
}
