from aijack_secureboost import SecureBoostClassifier as _SecureBoostClassifier


class SecureBoostClassifier(_SecureBoostClassifier):
    def __init__(self, *args, **kwargs):
        super(SecureBoostClassifier, self).__init__(*args, **kwargs)
