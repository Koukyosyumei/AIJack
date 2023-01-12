#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <cmath>
#include <vector>
#include <limits>
#include <cfenv>

using namespace std;
namespace py = pybind11;

constexpr double pi = 3.14159265358979323846;

double binom(double n, double k)
{
    return boost::math::binomial_coefficient<double>(n, k);
}

double _log_add(double logx, double logy)
{
    double a = min(logx, logy);
    double b = max(logx, logy);

    if (std::isinf(-1 * a))
    {
        return b;
    }
    return std::log(1 + std::exp(a - b)) + b;
}

double _log_sub(double logx, double logy)
{
    if (std::isinf(-1 * logy))
    {
        return logx;
    }
    if (logx == logy)
    {
        return -1 * std::numeric_limits<double>::infinity();
    }
    double result = std::log(std::exp((logx - logy) - 1)) + logy;
    if (std::fetestexcept(FE_OVERFLOW))
    {
        return logx;
    }
    else
    {
        return result;
    }
}

double _log_erfc(double x)
{
    return log1p(-boost::math::erf(x));
}

double logsumexp(const std::vector<double> &arr, const std::vector<int> &signs)
{
    double maxVal = arr[0];
    double sum = 0;
    size_t arr_size = arr.size();
    for (size_t i = 1; i < arr_size; i++)
    {
        if (arr[i] > maxVal)
        {
            maxVal = arr[i];
        }
    }

    for (size_t i = 0; i < arr_size; i++)
    {
        sum += signs[i] * std::exp(arr[i] - maxVal);
    }
    return std::log(sum) + maxVal;
}
