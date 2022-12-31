from ...manager import BaseManager


def attach_history_attack_to_client(cls, lam):
    class HistoryAttackClientWrapper(cls):
        """Implementation of history attack proposed in https://arxiv.org/pdf/2203.08669.pdf"""

        def __init__(self, *args, **kwargs):
            super(HistoryAttackClientWrapper, self).__init__(*args, **kwargs)

        def upload_gradients(self):
            """Upload the local gradients"""
            gradients = []
            for param, prev_param in zip(self.model.parameters(), self.prev_parameters):
                gradients.append((param - prev_param) * lam)
            return gradients

    return HistoryAttackClientWrapper


class HistoryAttackWrapper(BaseManager):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def attach(self, cls):
        return attach_history_attack_to_client(cls, *self.args, **self.kwargs)
