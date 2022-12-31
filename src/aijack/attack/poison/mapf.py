from ...manager import BaseManager


def attach_mapf_to_client(cls, lam, base_model_parameters):
    class MAPFClientWrapper(cls):
        """Implementation of MAPF proposed in https://arxiv.org/pdf/2203.08669.pdf"""

        def __init__(self, *args, **kwargs):
            super(MAPFClientWrapper, self).__init__(*args, **kwargs)

        def upload_gradients(self):
            """Upload the local gradients"""
            gradients = []
            for param, base_param in zip(
                self.model.parameters(), base_model_parameters
            ):
                gradients.append((base_param - param) * lam)
            return gradients

    return MAPFClientWrapper


class MAPFWrapper(BaseManager):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def attach(self, cls):
        return attach_mapf_to_client(cls, *self.args, **self.kwargs)
