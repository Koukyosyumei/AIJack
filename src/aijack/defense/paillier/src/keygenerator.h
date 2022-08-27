#pragma once
#include <unordered_map>
#include <random>
#include <cmath>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <cassert>
#include <boost/math/special_functions/round.hpp>
#include "paillier.h"
#include "prime.h"
using namespace std;

struct PaillierKeyGenerator
{
    int bit_size;

    PaillierKeyGenerator(int bit_size_ = 512)
    {
        bit_size = bit_size_;
    }

    pair<PaillierPublicKey, PaillierSecretKey> generate_keypair()
    {
        boost::random::random_device rng;

        Bint p = generate_probably_prime(bit_size);
        Bint q = generate_probably_prime(bit_size);

        if (p == q)
        {
            return generate_keypair();
        }

        Bint n = p * q;
        Bint n2 = n * n;
        boost::random::uniform_int_distribution<Bint> distr = boost::random::uniform_int_distribution<Bint>(0, n2 - 1);

        Bint g, lam, l_g2lam_mod_n2, mu;
        do
        {
            // g = distr(rng);
            g = n + 1;
            lam = lcm(p - 1, q - 1);
            l_g2lam_mod_n2 = L(modpow(g, lam, n * n), n);

        } while ((gcd(g, n2) != 1) && (gcd(l_g2lam_mod_n2, n) != 1));

        mu = boost::integer::mod_inverse(l_g2lam_mod_n2, n);

        PaillierPublicKey pk = PaillierPublicKey(n, g, n2);
        PaillierSecretKey sk = PaillierSecretKey(p, q, n, g, n2, lam, mu);

        return make_pair(pk, sk);
    }
};
