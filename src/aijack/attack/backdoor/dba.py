from ...manager import BaseManager


def attach_dba_to_client(cls):
    class DistributedBackdoorAttackClientWrapper(cls):
        def __init__(self, *args, **kwargs):
            super(DistributedBackdoorAttackClientWrapper, self).__init__(
                *args, **kwargs
            )


class DistributedBackdoorAttackManager(BaseManager):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def attach(self, cls):
        return attach_dba_to_client(cls, *self.args, **self.kwargs)
