#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <cmath>
#include <iostream>
#include <vector>

#include "rdp.cpp"
#include "search.cpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(aijack_dp_core, m)
{
      m.doc() = R"pbdoc(
        core of diferential_privacy
    )pbdoc";

      m.def("eps_gaussian",
            &eps_gaussian, R"pbdoc(eps_gaussian)pbdoc");

      m.def("eps_laplace",
            &eps_laplace, R"pbdoc(eps_laplace)pbdoc");

      m.def("eps_randresp",
            &eps_randresp, R"pbdoc(eps_randresp)pbdoc");

      m.def("culc_tightupperbound_lowerbound_of_rdp_with_theorem6and8_of_zhu_2019",
            &culc_tightupperbound_lowerbound_of_rdp_with_theorem6and8_of_zhu_2019,
            R"pbdoc(culc_tightupperbound_lowerbound_of_rdp_with_theorem6and8_of_zhu_2019)pbdoc");

      m.def("culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism",
            &culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism,
            R"pbdoc(culc_upperbound_of_rdp_with_Sampled_Gaussian_Mechanism)pbdoc");

      m.def("_ternary_search",
            &_ternary_search, R"pbdoc(_ternary_search)pbdoc");

      m.def("_ternary_search_int",
            &_ternary_search_int, R"pbdoc(_ternary_search_int)pbdoc");

      m.def("_greedy_search",
            &_greedy_search, R"pbdoc(_greedy_search)pbdoc");

      m.def("_greedy_search_frac",
            &_greedy_search_frac, R"pbdoc(_greey_search_frac)pbdoc");

#ifdef VERSION_INFO
      m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
      m.attr("__version__") = "dev";
#endif
}
