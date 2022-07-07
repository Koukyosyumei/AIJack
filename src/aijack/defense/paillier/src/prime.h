#pragma once
#include <cmath>
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
using namespace std;

namespace mp = boost::multiprecision;
using Bint = mp::cpp_int;
using Bfloat = mp::cpp_dec_float_50;

inline Bint gcd(Bint a, Bint b)
{
    if (a % b == 0)
    {
        return b;
    }
    else
    {
        return gcd(b, a % b);
    }
}

inline Bint lcm(Bint a, Bint b)
{
    return abs(a) / gcd(a, b) * abs(b);
}

inline Bint modpow(Bint x, Bint n, Bint m)
{
    // returns x^n (mod m)
    Bint ret = 1;
    while (n > 0)
    {
        if (n & 1)
            ret = ret * x % m;
        x = x * x % m;
        n >>= 1;
    }
    return ret;
}

inline bool cond_of_miller_rabin(Bint d, Bint a, Bint n)
{
    Bint t = d;
    Bint y = modpow(a, t, n);

    while ((t != n - 1) && (y != 1) && (y != n - 1))
    {
        y = (y * y) % n;
        t <<= 1;
    }

    return (y != n - 1) && (t % 2) == 0;
}

inline bool miller_rabin_primality_test(Bint n, Bint k = 40)
{
    boost::random::random_device rng;

    if (n <= 0)
    {
        return false;
    }

    if (n == 2)
    {
        return true;
    }

    if (n == 1 || n % 2 == 0)
    {
        return false;
    }

    Bint d = n - 1;
    Bint s = 0;
    while ((d % 2 == 0))
    {
        d /= 2;
        s += 1;
    }

    Bint nm1 = n - 1;
    boost::random::uniform_int_distribution<Bint> distr(1, n - 1);

    if (n < 2047)
    {
        for (Bint a : {2})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else if (n < Bint(1373653))
    {
        for (Bint a : {2, 3})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else if (n < Bint(9080191))
    {
        for (Bint a : {31, 73})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else if (n < Bint(25326001))
    {
        for (Bint a : {2, 3, 5})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else if (n < Bint(3215031751))
    {
        for (Bint a : {2, 3, 5, 7})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else if (n < Bint(4759123141))
    {
        for (Bint a : {2, 7, 61})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else if (n < Bint(2152302898747))
    {
        for (Bint a : {2, 3, 5, 7, 11})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else if (n < Bint(3474749660383))
    {
        for (Bint a : {2, 3, 5, 7, 11, 13})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else if (n < Bint(341550071728321))
    {
        for (Bint a : {2, 3, 5, 7, 11, 13, 17})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else
    {
        Bint a;
        for (Bint i = 0; i < k; i++)
        {
            a = distr(rng);
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    return true;
}

inline Bint generate_probably_prime(int bits_size)
{
    boost::random::random_device rng;

    Bint min_val = mp::pow(Bint(2), bits_size - 1);
    Bint max_val = min_val * 2 - 1;
    boost::random::uniform_int_distribution<Bint> distr(min_val, max_val);
    Bint p = 0;
    while (!miller_rabin_primality_test(p))
    {
        p = distr(rng);
    }
    return p;
}
