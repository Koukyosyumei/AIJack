#include <cmath>
#include <algorithm>
#include <iostream>
#include <functional>

using namespace std;

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

int main()
{
    std::cout << binom(3, 2) << std::endl;
    std::cout << binom(43, 23) << std::endl;
    std::cout << binom(-3, 2) << std::endl;
    std::cout << binom(-3.1, 2.2) << std::endl;
    std::cout << binom(2.2, 3.1) << std::endl;
    std::cout << binom(0.277464, 20.000000) << std::endl;

    std::cout <<
}