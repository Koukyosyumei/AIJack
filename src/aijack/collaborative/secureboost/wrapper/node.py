from aijack_secureboost import Node as _Node  # noqa:F401
from aijack_secureboost import Party as _Party  # noqa:F401


class Party(_Party):
    def __init__(self, *args, **kwargs):
        super(Party, self).__init__(*args, **kwargs)


class Node(_Node):
    def __init__(self, *args, **kwargs):
        super(Node, self).__init__(*args, **kwargs)
