#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <complex>
#include <cmath>
#include <vector>
#include <limits>
#include <cfenv>

using namespace std;
namespace py = pybind11;

constexpr double pi = 3.14159265358979323846;

double beta(double a, double b)
{
    return std::tgamma(a) * std::tgamma(b) / std::tgamma(a + b);
}

double lbeta(double a, double b)
{
    return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}

inline double binom(double n, double k)
{
    double kx, nx, num, den, dk, sgn;
    int i;

    if (n < 0)
    {
        nx = std::floor(n);
        if (n == nx)
        {
            // undefined
            return NAN;
        }
    }

    kx = std::floor(k);
    if (k == kx && (std::fabs(n) > 1e-8 || n == 0))
    {
        // Integer case: use multiplication formula for less rounding error
        // for cases where the result is an integer.
        //
        // This cannot be used for small nonzero n due to loss of
        // precision.

        nx = std::floor(n);
        if (nx == n && kx > nx / 2 && nx > 0)
        {
            // Reduce kx by symmetry
            kx = nx - kx;
        }

        if (kx >= 0 && kx < 20)
        {
            num = 1.0;
            den = 1.0;
            for (i = 1; i <= (int)kx; i++)
            {
                num *= i + n - kx;
                den *= i;
                if (std::fabs(num) > 1e50)
                {
                    num /= den;
                    den = 1.0;
                }
            }
            return num / den;
        }
    }

    // general case:
    if (n >= 1e10 * k && k > 0)
    {
        // avoid under/overflows in intermediate results
        return std::exp(-lbeta(1 + n - k, 1 + k) - std::log(n + 1));
    }
    else if (k > 1e8 * std::fabs(n))
    {
        // avoid loss of precision
        num = std::tgamma(1 + n) / std::fabs(k) + std::tgamma(1 + n) * n / (2 * k * k); // + ...
        num /= M_PI * std::fabs(k) * n;
        if (k > 0)
        {
            kx = std::floor(k);
            if ((int)kx == kx)
            {
                dk = k - kx;
                sgn = (int)kx % 2 == 0 ? 1 : -1;
            }
            else
            {
                dk = k;
                sgn = 1;
            }
            return num * std::sin((dk - n) * M_PI) * sgn;
        }
        else
        {
            kx = std::floor(k);
            if ((int)kx == kx)
            {
                return 0;
            }
            else
            {
                return num * std::sin(k * M_PI);
            }
        }
    }
    else
    {
        return 1 / (n + 1) / beta(1 + n - k, 1 + k);
    }
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
    double r = std::erfc(x);
    if (r == 0.)
    {
        return -1 * std::log(pi) / 2 -
               std::log(x) - pow(x, 2) -
               (0.5 * pow(x, -2)) +
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
