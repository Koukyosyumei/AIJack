import copy

import torch

from ...manager import BaseManager


def attach_freerider_to_client(cls, mu, sigma):
    class FreeRiderClientWrapper(cls):
        """Implementation of Free Rider Attack (https://arxiv.org/abs/1911.12560)"""

        def __init__(self, *args, **kwargs):
            super(FreeRiderClientWrapper, self).__init__(*args, **kwargs)
            self.prev_parameters_to_generate_fake_gradients = None

        def upload_gradients(self):
            """Upload the local gradients"""
            gradients = []
            if self.prev_parameters_to_generate_fake_gradients is not None:
                for param, prev_param in zip(
                    self.model.parameters(),
                    self.prev_parameters_to_generate_fake_gradients,
                ):
                    gradients.append(
                        (prev_param - param) / self.lr
                        + (sigma * torch.randn_like(param) + mu)
                    )
            else:
                for param in self.model.parameters():
                    gradients.append(sigma * torch.randn_like(param) + mu)
            return gradients

        def download(self, new_global_model):
            """Download the new global model"""
            self.prev_parameters_to_generate_fake_gradients = copy.deepcopy(
                self.model.parameters()
            )
            super().download(self, new_global_model)

    return FreeRiderClientWrapper


class FreeRiderClientManager(BaseManager):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def attach(self, cls):
        return attach_freerider_to_client(cls, *self.args, **self.kwargs)
