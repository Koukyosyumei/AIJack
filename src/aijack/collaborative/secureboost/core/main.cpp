#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <cmath>
#include <iostream>
#include <vector>

#include "secureboost.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(aijack_secureboost, m)
{
    m.doc() = R"pbdoc(
        core of SecureBoost
    )pbdoc";

    py::class_<Party>(m, "Party")
        .def(py::init<vector<vector<double>>, vector<int>, int, int, double>());

    py::class_<XGBoostClassifier>(m, "XGBoostClassifier")
        .def(py::init<double, double, int, int, double, int, double, double, double>())
        .def("fit", &XGBoostClassifier::fit)
        .def("get_grad", &XGBoostClassifier::get_grad)
        .def("get_hess", &XGBoostClassifier::get_hess)
        .def("get_init_pred", &XGBoostClassifier::get_init_pred)
        .def("predict_raw", &XGBoostClassifier::predict_raw)
        .def("predict_proba", &XGBoostClassifier::predict_proba);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
