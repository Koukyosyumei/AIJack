import numpy as np


def test_paillier_core():
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


def test_paillier_torch():
    import torch  # noqa: F401

    from aijack.defense.paillier import (  # noqa: F401
        PaillierKeyGenerator,
        PaillierTensor,
    )

    keygenerator = PaillierKeyGenerator(512)
    pk, sk = keygenerator.generate_keypair()

    ct_1 = pk.encrypt(13)
    ct_2 = pk.encrypt(0.5)
    ct_3 = ct_1 + ct_2

    pt_1 = PaillierTensor([ct_1, ct_2, ct_3])
    torch.testing.assert_allclose(
        pt_1.decypt(sk), torch.Tensor([13, 0.5, 13.5]), atol=1e-5, rtol=1
    )

    pt_2 = pt_1 + torch.Tensor([0.4, 0.1, 0.2])
    torch.testing.assert_allclose(
        pt_2.decypt(sk), torch.Tensor([13.4, 0.6, 13.7]), atol=1e-5, rtol=1
    )

    pt_3 = pt_1 * torch.Tensor([1, 2.5, 0.5])
    torch.testing.assert_allclose(
        pt_3.decypt(sk), torch.Tensor([13, 1.25, 6.75]), atol=1e-5, rtol=1
    )

    pt_4 = pt_1 - torch.Tensor([0.7, 0.3, 0.6])
    torch.testing.assert_allclose(
        pt_4.decypt(sk), torch.Tensor([14.3, 0.2, 12.9]), atol=1e-5, rtol=1
    )

    pt_5 = pt_1 * 2
    torch.testing.assert_allclose(
        pt_5.decypt(sk), torch.Tensor([26, 1, 27]), atol=1e-5, rtol=1
    )
