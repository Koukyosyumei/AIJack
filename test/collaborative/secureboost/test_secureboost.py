def test_secureboostclassifier():
    import numpy as np  # noqa: F401

    from aijack.collaborative.secureboost import (  # noqa: F401
        Party,
        SecureBoostClassifier,
    )

    min_leaf = 1
    depth = 3
    learning_rate = 0.4
    boosting_rounds = 2
    lam = 1.0
    gamma = 0.0
    eps = 1.0
    min_child_weight = -1 * float("inf")
    subsample_cols = 1.0

    clf = SecureBoostClassifier(
        subsample_cols,
        min_child_weight,
        depth,
        min_leaf,
        learning_rate,
        boosting_rounds,
        lam,
        gamma,
        eps,
    )

    x1 = [12, 32, 15, 24, 20, 25, 17, 16]
    x1 = [[x] for x in x1]
    x2 = [1, 1, 0, 0, 1, 1, 0, 1]
    x2 = [[x] for x in x2]
    y = [1, 0, 1, 0, 1, 1, 0, 1]

    p1 = Party(x1, [0], 0, min_leaf, subsample_cols)
    p2 = Party(x2, [1], 1, min_leaf, subsample_cols)
    parties = [p1, p2]

    base_pred = clf.get_init_pred(y)
    assert np.allclose(base_pred, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    assert np.allclose(
        clf.get_grad(base_pred, y),
        np.array(
            [
                -0.26894,
                0.73106,
                -0.26894,
                0.73106,
                -0.26894,
                -0.26894,
                0.73106,
                -0.26894,
            ]
        ),
    )

    assert np.allclose(
        clf.get_hess(base_pred, y),
        np.array(
            [0.19661, 0.19661, 0.19661, 0.19661, 0.19661, 0.19661, 0.19661, 0.19661]
        ),
    )

    clf.fit(parties, y)

    X = [[12, 1], [32, 1], [15, 0], [24, 0], [20, 1], [25, 1], [17, 0], [16, 1]]

    assert np.allclose(
        clf.predict_raw(X),
        np.array(
            [
                1.38379341,
                0.53207456,
                1.38379341,
                0.22896408,
                1.29495549,
                1.29495549,
                0.22896408,
                1.38379341,
            ]
        ),
    )

    assert np.allclose(
        clf.predict_proba(X),
        np.array(
            [
                0.79959955,
                0.62996684,
                0.79959955,
                0.55699226,
                0.78498478,
                0.78498478,
                0.55699226,
                0.79959955,
            ]
        ),
    )
