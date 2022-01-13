#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <numbers>
#include <cmath>
#include <vector>
#include <limits>

using namespace std;
namespace py = pybind11;

double binom(double n, double k)
{
    return 1 / ((n + 1) * std::beta(n - k + 1, k + 1));
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
    return std::log(std::exp((logx - logy) - 1)) + logy;
}

double _log_erfc(double x)
{
    double r = std::erfc(x);
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
