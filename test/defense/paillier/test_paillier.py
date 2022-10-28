import numpy as np


def test_paillier():
    from aijack.defense.paillier import PaillierKeyGenerator  # noqa: F401

    keygenerator = PaillierKeyGenerator(512)
    pk, sk = keygenerator.generate_keypair()

    ct_1 = pk.encrypt(13)
    assert sk.decrypt2int(ct_1) == 13

    ct_2 = ct_1 * 2
    assert sk.decrypt2int(ct_2) == 26

    ct_3 = ct_1 + 5.6
    np.testing.assert_array_almost_equal(sk.decrypt2float(ct_3), 18.6, decimal=6)

    ct_4 = ct_1 + ct_3
    np.testing.assert_array_almost_equal(sk.decrypt2float(ct_4), 31.6, decimal=6)
