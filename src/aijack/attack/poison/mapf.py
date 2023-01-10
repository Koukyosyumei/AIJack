import torch

from ...manager import BaseManager


def attach_mapf_to_client(cls, lam, base_model_parameters=None):
    class MAPFClientWrapper(cls):
        """Implementation of MAPF proposed in https://arxiv.org/pdf/2203.08669.pdf"""

        def __init__(self, *args, **kwargs):
            super(MAPFClientWrapper, self).__init__(*args, **kwargs)

            if base_model_parameters is None:
                self.base_model_parameters = [
                    torch.randn_like(p) for p in self.model.parameters()
                ]
            else:
                self.base_model_parameters = base_model_parameters

        def upload_gradients(self):
            """Upload the local gradients"""
            gradients = []
            for param, base_param in zip(
                self.model.parameters(), self.base_model_parameters
            ):
                gradients.append((base_param - param) * lam)
            return gradients

    return MAPFClientWrapper


class MAPFClientWrapper(BaseManager):
    def attach(self, cls):
        return attach_mapf_to_client(cls, *self.args, **self.kwargs)
