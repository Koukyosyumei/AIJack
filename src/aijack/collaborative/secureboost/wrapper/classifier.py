from aijack_secureboost import SecureBoostClassifier as _SecureBoostClassifier


class SecureBoostClassifier(_SecureBoostClassifier):
    def __init__(
        self,
        subsample_cols=1.0,
        min_child_weight=-1 * float("inf"),
        depth=3,
        min_leaf=1,
        learning_rate=0.4,
        boosting_rounds=2,
        lam=1.0,
        gamma=0.0,
        eps=1.0,
    ):
        super(SecureBoostClassifier, self).__init__(
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
