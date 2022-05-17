from aijack_secureboost import Party as _Party  # noqa:F401


class Party(_Party):
    def __init__(self, *args, **kwargs):
        super(Party, self).__init__(*args, **kwargs)
