from ...manager import BaseManager


def attach_history_attack_to_client(cls, lam):
    """Attaches a history attack to a client.

    Args:
        cls: The client class.
        lam (float): The lambda parameter for the attack.

    Returns:
        class: A wrapper class with attached history attack.
    """

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


class HistoryAttackClientWrapper(BaseManager):
    def attach(self, cls):
        return attach_history_attack_to_client(cls, *self.args, **self.kwargs)
