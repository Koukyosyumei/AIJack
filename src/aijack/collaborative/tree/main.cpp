#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <cmath>
#include <iostream>
#include <vector>

#include "xgboost/xgboost.h"
#include "secureboost/secureboost.h"
#include "secureboost/mpisecureboost.h"
#include "../../defense/paillier/src/paillier.h"
#include "../../defense/paillier/src/keygenerator.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace std;
namespace py = pybind11;

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

PYBIND11_MODULE(aijack_secureboost, m)
{
    m.doc() = R"pbdoc(
        core of XGBoost
    )pbdoc";

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

    py::class_<MPISecureBoostParty>(m, "MPISecureBoostParty");

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

    py::class_<MPISecureBoostClassifier>(m, "MPISecureBoostClassifier");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
