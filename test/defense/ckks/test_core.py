def test_encode():
    import numpy as np
    from secure_ml.defense import CKKS

    M = 8
    scale = 1 << 20
    ckks = CKKS(M, scale)

    b = np.array([1, 2, 3, 4])
    p = ckks.sigma_inverse(b)
    b_reconstructed = ckks.sigma(p)
    np.testing.assert_array_almost_equal(b, b_reconstructed, decimal=6)

    m1 = np.array([1, 2, 3, 4])
    m2 = np.array([1, -2, 3, -4])
    p1 = ckks.sigma_inverse(m1)
    p2 = ckks.sigma_inverse(m2)
    p_add = p1 + p2
    p_mult = p1 * p2
    np.testing.assert_array_almost_equal(m1 + m2, ckks.sigma(p_add), decimal=6)
    np.testing.assert_array_almost_equal(m1 * m2, ckks.sigma(p_mult), decimal=6)

    z = np.array(
        [
            3 + 4j,
            2 - 1j,
        ]
    )
    p = ckks.encode(z)
    np.testing.assert_array_almost_equal(z, ckks.decode(p), decimal=6)
