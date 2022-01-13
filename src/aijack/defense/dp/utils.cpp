#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <numbers>
#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

using namespace std;
namespace py = pybind11;

long double binom(long double n, long double k)
{
    return 1 / ((n + 1) * std::beta(n - k + 1, k + 1));
}

long double _log_add(long double logx, long double logy)
{
    long double a = min(logx, logy);
    long double b = max(logx, logy);

    if (std::isinf(-1 * a))
    {
        return b;
    }
    return std::log(1 + std::exp(a - b)) + b;
}

long double _log_sub(long double logx, long double logy)
{
    if (std::isinf(-1 * logy))
    {
        return logx;
    }
    if (logx == logy)
    {
        return -1 * std::numeric_limits<long double>::infinity();
    }
    return std::log(std::exp((logx - logy) - 1)) + logy;
}

long double _log_erfc(long double x)
{
    long double r = std::erfc(x);
    if (r == 0.)
    {
        return std::log(std::numbers::pi) / 2 -
               std::log(x) - pow(x, 2) -
               (0.5 * pow(x, -1)) +
               0.625 * pow(x, -4) -
               37.0 / 24.0 * pow(x, -6) +
               353.0 / 64.0 * pow(x, -8);
    }
    else
    {
        return std::log(r);
    }
}

long double logsumexp(const std::vector<long double> &arr, const std::vector<int> &signs)
{
    long double maxVal = arr[0];
    long double sum = 0;
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
