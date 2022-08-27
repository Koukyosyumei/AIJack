#pragma once
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_free.hpp>
#include "paillier.h"
using namespace std;

namespace boost
{
    namespace serialization
    {

        template <class Archive>
        void serialize(Archive &ar, PaillierPublicKey &pk, unsigned int /* version */)
        {
            ar &make_nvp("n", pk.n);
            ar &make_nvp("n2", pk.n2);
            ar &make_nvp("g", pk.g);
            ar &make_nvp("precision", pk.precision);
            ar &make_nvp("max_val", pk.max_val);
        }

        template <class Archive>
        void serialize(Archive &ar, PaillierCipherText &ct, unsigned int /* version */)
        {
            ar &make_nvp("pk", ct.pk);
            ar &make_nvp("c", ct.c);
            ar &make_nvp("exponent", ct.exponent);
            ar &make_nvp("precision", ct.precision);
        }

        template <class Archive>
        void serialize(Archive &ar, pair<PaillierCipherText, PaillierCipherText> &pair_ct, unsigned int /* version */)
        {
            ar &make_nvp("first_pk", pair_ct.first.pk);
            ar &make_nvp("first_c", pair_ct.first.c);
            ar &make_nvp("first_exponent", pair_ct.first.exponent);
            ar &make_nvp("first_precision", pair_ct.first.precision);

            ar &make_nvp("second_pk", pair_ct.second.pk);
            ar &make_nvp("second_c", pair_ct.second.c);
            ar &make_nvp("second_exponent", pair_ct.second.exponent);
            ar &make_nvp("second_precision", pair_ct.second.precision);
        }
    }
}
