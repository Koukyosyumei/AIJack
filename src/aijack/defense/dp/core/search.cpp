#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vector>
#include <cmath>
#include <numeric>

using namespace std;
namespace py = pybind11;

double _ternary_search(const std::function<double(double)> &f,
                       double left, double right,
                       std::vector<double> orders,
                       double precision, int max_iteration = 10000)
{
    int i = 0;
    double left_third, right_third;
    while (i < max_iteration)
    {
        if (abs(right - left) < precision)
        {
            return (left + right) / 2;
        }

        left_third = left + (right - left) / 3;
        right_third = right - (right - left) / 3;

        if (f(left_third) < f(right_third))
        {
            right = right_third;
        }
        else
        {
            left = left_third;
        }
        i++;
    }
    return (left + right) / 2;
}

int _ternary_search_int(const std::function<double(int)> &f,
                        int left, int right,
                        std::vector<int> orders,
                        double precision, int max_iteration = 10000)
{
    int lo = left;
    int hi = right;

    int i = 0;
    int mid;
    while ((hi - lo > 1) && (i < max_iteration))
    {
        mid = (hi + lo) >> 1;
        if (f(mid) > f(mid + 1))
        {
            hi = mid;
        }
        else
        {
            lo = mid;
        }
        i += 1;
    }
    return lo + 1;
}

int _greedy_search(const std::function<double(int)> &f,
                   int left, int right,
                   std::vector<int> orders,
                   double precision, int max_iteration = 10000)
{
    double min_val = std::numeric_limits<double>::infinity();
    int optim_lam = left;
    if (orders.size() == 0)
    {
        orders = std::vector<int>(right - left + 1);
        std::iota(orders.begin(), orders.end(), 0);
    }

    for (auto lam : orders)
    {
        double val = f(lam);
        if (min_val > val)
        {
            optim_lam = lam;
            min_val = val;
        }
    }

    return optim_lam;
}

double _greedy_search_frac(const std::function<double(double)> &f,
                           double left, double right,
                           std::vector<double> orders,
                           double precision, int max_iteration = 10000)
{
    double min_val = std::numeric_limits<double>::infinity();
    double optim_lam = left;
    if (orders.size() == 0)
    {
        return 0;
    }

    for (auto lam : orders)
    {
        double val = f(lam);
        if (min_val > val)
        {
            optim_lam = lam;
            min_val = val;
        }
    }

    return optim_lam;
}
