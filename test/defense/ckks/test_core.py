def test_sigma():
    import numpy as np
    from secure_ml.defense import CKKSEncoder

    M = 8
    scale = 1 << 20
    encoder = CKKSEncoder(M, scale)

    b = np.array([1, 2, 3, 4])
    p = encoder.sigma_inverse(b)
    b_reconstructed = encoder.sigma(p)
    np.testing.assert_array_almost_equal(b, b_reconstructed, decimal=6)

    m1 = np.array([1, 2, 3, 4])
    m2 = np.array([1, -2, 3, -4])
    p1 = encoder.sigma_inverse(m1)
    p2 = encoder.sigma_inverse(m2)
    p_add = p1 + p2
    p_mult = p1 * p2
    np.testing.assert_array_almost_equal(m1 + m2, encoder.sigma(p_add), decimal=6)
    np.testing.assert_array_almost_equal(m1 * m2, encoder.sigma(p_mult), decimal=6)


def test_encoder():
    import numpy as np
    from secure_ml.defense import CKKSEncoder

    M = 8
    scale = 1 << 20
    encoder = CKKSEncoder(M, scale)

    z1 = np.array(
        [
            3 + 4j,
            2 - 1j,
        ]
    )
    p1 = encoder.encode(z1)
    z2 = np.array(
        [
            5 + 2j,
            1 - 6j,
        ]
    )
    p2 = encoder.encode(z2)
    np.testing.assert_array_almost_equal(z1, encoder.decode(p1), decimal=6)
    p_add = p1 + p2
    p_mult = p1 * p2
    np.testing.assert_array_almost_equal(z1 + z2, encoder.decode(p_add), decimal=6)
    np.testing.assert_array_almost_equal(z1 * z2, encoder.decode(p_mult), decimal=4)
