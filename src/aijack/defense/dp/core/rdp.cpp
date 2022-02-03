#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <cmath>
#include <vector>
#include "utils.cpp"

using namespace std;
namespace py = pybind11;

double eps_gaussian(double alpha, py::dict params)
{
    if (std::isinf(alpha))
    {
        return min(4 * std::exp(eps_gaussian(2, params) - 1),
                   2 * std::exp(eps_gaussian(2, params)));
    }
    return alpha / (2 * pow(params["sigma"].cast<double>(), 2));
}

double eps_laplace(double alpha, py::dict params)
{
    double b = params["b"].cast<double>();
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
        std::vector<double> val(2);
        std::vector<int> sgn(2);
        val[0] = std::log(alpha / (2 * alpha - 1)) + (alpha - 1) / b;
        sgn[0] = 1;
        val[1] = std::log((alpha - 1) / (2 * alpha - 1)) + (-1 * alpha) / b;
        sgn[1] = 1;
        return (1 / (alpha - 1)) * logsumexp(val, sgn);
    }
}

double eps_randresp(double alpha, py::dict params)
{
    double inf = std::numeric_limits<double>::infinity();
    double p = params["p"].cast<double>();
    if (p == 1 || p == 0)
    {
        return inf;
    }
    if (alpha <= 1)
    {
        return (2 * p - 1) * std::log(p / (1 - p));
    }
    else if (alpha == inf)
    {
        std::abs(std::log(1.0 * p / (1 - p)));
    }

    std::vector<double> terms(2);
    std::vector<int> signs(2);
    terms[0] = alpha * std::log(p) + (1 - alpha) * std::log(1 - p);
    signs[0] = 1;
    terms[1] = alpha * std::log(1 - p) + (1 - alpha) * p;
    signs[1] = 1;
    return (1 / (alpha - 1)) * logsumexp(terms, signs);
}

double culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism_int(int alpha,
                                                                  py::dict params,
                                                                  double sampling_rate)
{
    double sigma = params["sigma"].cast<double>();
    double inv_double_sigma_square = 1 / (2 * (sigma * sigma));
    double log_a = -1 * std::numeric_limits<double>::infinity();

    for (int i = 0; i <= alpha; i++)
    {
        double log_coef_i = std::log(binom(alpha, i)) +
                            i * std::log(sampling_rate) +
                            (alpha - i) * std::log(1 - sampling_rate);

        double s = log_coef_i + (i * i - i) * inv_double_sigma_square;
        log_a = _log_add(log_a, s);
    }
    return log_a / (alpha - 1);
}

double culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism_frac(double alpha,
                                                                   py::dict params,
                                                                   double sampling_rate)
{

    double sigma = params["sigma"].cast<double>();
    double inv_double_sigma_square = 1 / (2 * (sigma * sigma));
    double log_sampling_rate = std::log(sampling_rate);
    double log_1m_sampling_rate = std::log(1 - sampling_rate);
    double log_half = std::log(0.5);

    double log_a0 = -1 * std::numeric_limits<double>::infinity();
    double log_a1 = -1 * std::numeric_limits<double>::infinity();
    double i = 0;

    double z0 = pow(sigma, 2) * std::log(1 / sampling_rate - 1) + 0.5;

    double coef, log_coef, j, log_t0, log_t1, log_e0, log_e1, log_s0, log_s1;
    while (true)
    {
        coef = binom(alpha, i);

        if (std::isnan(coef))
        {
            auto warnings = py::module::import("warnings");
            warnings.attr("warn")(
                "Culculation of RDP did not converge. Please consider using backend=python");
            return std::nan("");
        }

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

double culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism(double alpha,
                                                              py::dict params,
                                                              double sampling_rate,
                                                              const std::function<double(double, py::dict)> &_eps)
{
    if (fmod(alpha, 1) == 0.0)
    {
        return culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism_int(
            (int)alpha, params, sampling_rate);
    }
    else
    {
        return culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism_frac(
            alpha, params, sampling_rate);
    }
}

double culc_tightupperbound_lowerbound_of_rdp_with_theorem6and8_of_zhu_2019(int alpha,
                                                                            py::dict params,
                                                                            double sampling_rate,
                                                                            const std::function<double(double, py::dict)> &_eps)
{
    std::vector<double> terms(alpha);
    std::vector<int> signs(alpha);

    double first = ((alpha - 1) * std::log(1 - sampling_rate)) +
                   std::log(alpha * sampling_rate - sampling_rate + 1);
    terms[0] = first;
    signs[0] = 1;

    if (alpha >= 2)
    {
        double second;
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

    return (1 / ((double)alpha - 1)) * logsumexp(terms, signs);
}
