#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <cmath>
#include <iostream>
#include <vector>

#include "paillier.h"
#include "keygenerator.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace std;
namespace py = pybind11;

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

PYBIND11_MODULE(aijack_paillier, m)
{
    m.doc() = R"pbdoc(
        core of Paillier Encryption Scheme
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

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
