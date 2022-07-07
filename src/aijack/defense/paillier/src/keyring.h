#pragma once
#include "paillier.h"
#include "../tsl/robin_map.h"
#include "../tsl/robin_set.h"
using namespace std;

struct HashPairSzudzikBint
{
    // implementation of szudzik paring
    template <class T1, class T2>
    size_t operator()(const pair<T1, T2> &p) const
    {
        size_t seed;
        if (p.first >= p.second)
        {
            seed = std::hash<T1>{}(p.first * p.first + p.first + p.second);
        }
        else
        {
            seed = std::hash<T1>{}(p.second * p.second + p.first);
        }
        return seed;
    }
};

struct PaillierKeyRing
{
    tsl::robin_map<pair<Bint, Bint>, PaillierSecretKey, HashPairSzudzikBint> keyring;

    PaillierKeyRing(){};

    void add(PaillierSecretKey sk)
    {
        keyring.emplace(make_pair(sk.n, sk.g), sk);
    }

    PaillierSecretKey get_sk(PaillierPublicKey pk)
    {
        return keyring[make_pair(pk.n, pk.g)];
    }

    template <typename T>
    T decrypt(PaillierCipherText pt);
};

template <typename T>
inline T PaillierKeyRing::decrypt(PaillierCipherText pt)
{
    return get_sk(pt.pk).decrypt<T>(pt);
}
