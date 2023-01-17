#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <cmath>
#include <iostream>
#include <vector>

#include "aijack/defense/dp/core//rdp.cpp"
#include "aijack/defense/dp/core//search.cpp"
#include "aijack/defense/kanonymity/core/anonymizer.h"
#include "aijack/defense/paillier/src/paillier.h"
#include "aijack/defense/paillier/src/keygenerator.h"
#include "aijack/collaborative/tree/xgboost/xgboost.h"
#include "aijack/collaborative/tree/secureboost/secureboost.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace std;
namespace py = pybind11;

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

PYBIND11_MODULE(aijack_cpp_core, m)
{
    m.doc() = R"pbdoc(
        c++ backend for aijack
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

    py::class_<PaillierKeyGenerator>(m, "PaillierKeyGenerator")
        .def(py::init<int>())
        .def("generate_keypair", &PaillierKeyGenerator::generate_keypair);

    py::class_<PaillierPublicKey>(m, "PaillierPublicKey")
        .def("encrypt", &PaillierPublicKey::encrypt<int>)
        .def("encrypt", &PaillierPublicKey::encrypt<long>)
        .def("encrypt", &PaillierPublicKey::encrypt<float>)
        .def("encrypt", &PaillierPublicKey::encrypt<double>)
        .def("get_publickeyvalues", &PaillierPublicKey::get_publickeyvalues);

    py::class_<PaillierCipherText>(m, "PaillierCipherText")
        .def("__add__", overload_cast_<int>()(&PaillierCipherText::operator+))
        .def("__add__", overload_cast_<long>()(&PaillierCipherText::operator+))
        .def("__add__", overload_cast_<float>()(&PaillierCipherText::operator+))
        .def("__add__", overload_cast_<double>()(&PaillierCipherText::operator+))
        .def("__add__", overload_cast_<PaillierCipherText>()(&PaillierCipherText::operator+))
        .def("__mul__", overload_cast_<int>()(&PaillierCipherText::operator*))
        .def("__mul__", overload_cast_<long>()(&PaillierCipherText::operator*))
        .def("__mul__", overload_cast_<float>()(&PaillierCipherText::operator*))
        .def("__mul__", overload_cast_<double>()(&PaillierCipherText::operator*))
        .def("get_value", &PaillierCipherText::get_value);

    py::class_<PaillierSecretKey>(m, "PaillierSecretKey")
        .def("decrypt2int", &PaillierSecretKey::decrypt2int)
        .def("decrypt2long", &PaillierSecretKey::decrypt2long)
        .def("decrypt2float", &PaillierSecretKey::decrypt2float)
        .def("decrypt2double", &PaillierSecretKey::decrypt2double)
        .def("get_publickeyvalues", &PaillierSecretKey::get_publickeyvalues)
        .def("get_secretkeyvalues", &PaillierSecretKey::get_secretkeyvalues);

    py::class_<XGBoostParty>(m, "XGBoostParty")
        .def(py::init<vector<vector<float>>, int, vector<int>, int,
                      int, float, int, bool, int>())
        .def("get_lookup_table", &XGBoostParty::get_lookup_table);

    py::class_<SecureBoostParty>(m, "SecureBoostParty")
        .def(py::init<vector<vector<float>>, int, vector<int>, int,
                      int, float, int, bool, int>())
        .def("get_lookup_table", &SecureBoostParty::get_lookup_table)
        .def("set_publickey", &SecureBoostParty::set_publickey)
        .def("set_secretkey", &SecureBoostParty::set_secretkey);

    py::class_<XGBoostNode>(m, "XGBoostNode")
        .def("get_idxs", &XGBoostNode::get_idxs)
        .def("get_party_id", &XGBoostNode::get_party_id)
        .def("get_record_id", &XGBoostNode::get_record_id)
        .def("get_val", &XGBoostNode::get_val)
        .def("get_score", &XGBoostNode::get_score)
        .def("get_left", &XGBoostNode::get_left)
        .def("get_right", &XGBoostNode::get_right)
        .def("is_leaf", &XGBoostNode::is_leaf);

    py::class_<SecureBoostNode>(m, "SecureBoostNode")
        .def("get_idxs", &SecureBoostNode::get_idxs)
        .def("get_party_id", &SecureBoostNode::get_party_id)
        .def("get_record_id", &SecureBoostNode::get_record_id)
        .def("get_val", &SecureBoostNode::get_val)
        .def("get_score", &SecureBoostNode::get_score)
        .def("get_left", &SecureBoostNode::get_left)
        .def("get_right", &SecureBoostNode::get_right)
        .def("is_leaf", &SecureBoostNode::is_leaf);

    py::class_<XGBoostTree>(m, "XGBoostTree")
        .def("get_root_xgboost_node", &XGBoostTree::get_root_xgboost_node)
        .def("print", &XGBoostTree::print)
        .def("predict", &XGBoostTree::predict);

    py::class_<SecureBoostTree>(m, "SecureBoostTree")
        .def("print", &SecureBoostTree::print)
        .def("predict", &SecureBoostTree::predict);

    py::class_<XGBoostClassifier>(m, "XGBoostClassifier")
        .def(py::init<int, float, float, int, int, float, int, float, float, float,
                      int, int, float, int, bool>())
        .def("fit", &XGBoostClassifier::fit)
        .def("get_init_pred", &XGBoostClassifier::get_init_pred)
        .def("load_estimators", &XGBoostClassifier::load_estimators)
        .def("get_estimators", &XGBoostClassifier::get_estimators)
        .def("predict_raw", &XGBoostClassifier::predict_raw)
        .def("predict_proba", &XGBoostClassifier::predict_proba);

    py::class_<SecureBoostClassifier>(m, "SecureBoostClassifier")
        .def(py::init<int, float, float, int, int, float, int, float, float, float,
                      int, int, float, int, bool>())
        .def("fit", &SecureBoostClassifier::fit)
        .def("get_init_pred", &SecureBoostClassifier::get_init_pred)
        .def("load_estimators", &SecureBoostClassifier::load_estimators)
        .def("get_estimators", &SecureBoostClassifier::get_estimators)
        .def("predict_raw", &SecureBoostClassifier::predict_raw)
        .def("predict_proba", &SecureBoostClassifier::predict_proba);

    py::class_<DataFrame>(m, "DataFrame")
        .def(py::init<vector<string>, map<string, bool>, int>())
        .def("insert_continuous", &DataFrame::insert_continuous)
        .def("insert_categorical", &DataFrame::insert_categorical)
        .def("insert_continuous_column", &DataFrame::insert_continuous_column)
        .def("insert_categorical_column", &DataFrame::insert_categorical_column)
        .def("get_data_continuous", &DataFrame::get_data_continuous)
        .def("get_data_categorical", &DataFrame::get_data_categorical);

    py::class_<Mondrian>(m, "Mondrian")
        .def(py::init<int>())
        .def("anonymize", &Mondrian::anonymize);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
