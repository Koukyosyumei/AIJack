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
        .def(py::init<vector<vector<double>>, vector<int>, int, int, double>())
        .def("get_lookup_table", &Party::get_lookup_table)
        .def("get_percentiles", &Party::get_percentiles)
        .def("is_left", &Party::is_left)
        .def("greedy_search_split", &Party::greedy_search_split)
        .def("split_rows", &Party::split_rows)
        .def("insert_lookup_table", &Party::insert_lookup_table);

    py::class_<Node>(m, "Node")
        .def(py::init<vector<Party> &, vector<double>, vector<double>,
                      vector<double>, vector<int>,
                      double, double, double, double,
                      int>())
        .def("get_idxs", &Node::get_idxs)
        .def("get_party_id", &Node::get_party_id)
        .def("get_record_id", &Node::get_record_id)
        .def("get_val", &Node::get_val)
        .def("get_score", &Node::get_score)
        .def("get_left", &Node::get_left)
        .def("get_right", &Node::get_right)
        .def("get_parties", &Node::get_parties)
        .def("is_leaf", &Node::is_leaf)
        .def("print", &Node::print, py::arg("binary_color") = true);

    py::class_<XGBoostTree>(m, "XGBoostTree")
        .def("get_root_node", &XGBoostTree::get_root_node);

    py::class_<SecureBoostClassifier>(m, "SecureBoostClassifier")
        .def(py::init<double, double, int, int, double, int, double, double, double>())
        .def("fit", &SecureBoostClassifier::fit)
        .def("get_grad", &SecureBoostClassifier::get_grad)
        .def("get_hess", &SecureBoostClassifier::get_hess)
        .def("get_init_pred", &SecureBoostClassifier::get_init_pred)
        .def("load_estimators", &SecureBoostClassifier::load_estimators)
        .def("get_estimators", &SecureBoostClassifier::get_estimators)
        .def("predict_raw", &SecureBoostClassifier::predict_raw)
        .def("predict_proba", &SecureBoostClassifier::predict_proba);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
