#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <cmath>
#include <vector>
#include <limits>
#include <cfenv>

using namespace std;
namespace py = pybind11;

constexpr double pi = 3.14159265358979323846;

double binom(double n, double k)
{
    double kx, nx, num, den, dk, sgn;
    int i;

    if (n < 0)
    {
        nx = std::floor(n);
        if (n == nx)
        {
            return std::nan("");
        }
    }

    kx = std::floor(k);
    if ((k == kx) && (fabs(n) > 1e-8 or n == 0))
    {
        nx = std::floor(n);
        if (nx == n && kx > nx / 2 && nx > 0)
        {
            kx = nx - kx;
        }

        if (kx >= 0 && kx < 20)
        {
            num = 1.0;
            den = 1.0;
            for (int i = 1; i < 1 + (int)kx; i++)
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

    if ((n >= 1e10 * k) && (k > 0))
    {
        return std::exp(-std::log(std::beta(1 + n - k, 1 + k))) - std::log(n + 1);
    }
    else if (k > 1e8 * std::fabs(n))
    {
        num = std::tgamma(1 + n) / std::fabs(k) + std::tgamma(1 + n) * n / (2 * (k * k));
        num /= pi * pow(std::fabs(k), n);
        if (k > 0)
        {
            kx = std::floor(k);
            if ((int)kx == kx)
            {
                dk = k - kx;
                if ((int)kx % 2 == 0)
                {
                    sgn = 1;
                }
                else
                {
                    sgn = -1;
                }
            }
            else
            {
                dk = k;
                sgn = 1;
            }
            return num * std::sin((dk - n) * pi) * sgn;
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
                return num * std::sin(k * pi);
            }
        }
    }
    else
    {
        return 1 / (n + 1) / std::beta(1 + n - k, 1 + k);
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
