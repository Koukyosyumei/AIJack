import numpy as np


def test_secureboost():
    from aijack.collaborative.tree import (  # noqa:F401
        SecureBoostClassifierAPI,
        SecureBoostClient,
    )
    from aijack.defense.paillier import PaillierKeyGenerator  # noqa:F401

    min_leaf = 1
    depth = 3
    learning_rate = 0.4
    boosting_rounds = 2
    lam = 1.0
    gamma = 0.0
    eps = 1.0
    min_child_weight = -1 * float("inf")
    subsample_cols = 1.0

    x1 = [12, 32, 15, 24, 20, 25, 17, 16]
    x1 = [[x] for x in x1]
    x2 = [1, 1, 0, 0, 1, 1, 0, 1]
    x2 = [[x] for x in x2]
    y = [1, 0, 1, 0, 1, 1, 0, 1]

    keygenerator = PaillierKeyGenerator(512)
    pk, sk = keygenerator.generate_keypair()

    sp1 = SecureBoostClient(
        x1, 2, [0], 0, min_leaf, subsample_cols, 256, False, 0)
    sp2 = SecureBoostClient(
        x2, 2, [1], 1, min_leaf, subsample_cols, 256, False, 0)
    sparties = [sp1, sp2]

    sparties[0].set_publickey(pk)
    sparties[1].set_publickey(pk)
    sparties[0].set_secretkey(sk)

    sclf = SecureBoostClassifierAPI(
        2,
        subsample_cols,
        min_child_weight,
        depth,
        min_leaf,
        learning_rate,
        boosting_rounds,
        lam,
        gamma,
        eps,
        0,
        0,
        1.0,
        1,
        True,
    )
    sclf.fit(sparties, y)

    x_test = [[12, 1], [32, 1], [15, 0], [24, 0],
              [20, 1], [25, 1], [17, 0], [16, 1]]
    y_test_proba = sclf.predict_proba(x_test)

    y_test_proba_true = [
        [0.20040041208267212, 0.7995995879173279],
        [0.3700332045555115, 0.6299667954444885],
        [0.20040041208267212, 0.7995995879173279],
        [0.44300776720046997, 0.55699223279953],
        [0.2150152325630188, 0.7849847674369812],
        [0.2150152325630188, 0.7849847674369812],
        [0.44300776720046997, 0.55699223279953],
        [0.20040041208267212, 0.7995995879173279],
    ]

    assert np.array_equal(y_test_proba, y_test_proba_true)
