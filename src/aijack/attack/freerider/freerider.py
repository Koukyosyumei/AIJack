import copy

import torch

from ...manager import BaseManager


def attach_freerider_to_client(cls, mu, sigma):
    """Wraps the given class in FreeRiderClientWrapper.

    Args:
        mu (float): mean of the gaussian distribution used to generate fake gradients
        sigma (float): standard deviation of the gaussian distribution used to generate fake gradients

    Returns:
        cls: a class wrapped in FreeRiderClientWrapper
    """

    class FreeRiderClientWrapper(cls):
        """Implementation of Free Rider Attack (https://arxiv.org/abs/1911.12560)"""

        def __init__(self, *args, **kwargs):
            super(FreeRiderClientWrapper, self).__init__(*args, **kwargs)
            self.prev_parameters_to_generate_fake_gradients = None

        def upload_gradients(self):
            """Uploads the fake gradients"""
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
            """Downloads the new global model"""
            self.prev_parameters_to_generate_fake_gradients = copy.deepcopy(
                list(self.model.parameters())
            )
            super().download(new_global_model)

    return FreeRiderClientWrapper


class FreeRiderClientManager(BaseManager):
    """Manager class for Free-Rider Attack (https://arxiv.org/abs/1911.12560)"""

    def attach(self, cls):
        """Wraps the given class in FreeRiderClientWrapper.

        Returns:
            cls: a class wrapped in FreeRiderClientWrapper
        """
        return attach_freerider_to_client(cls, *self.args, **self.kwargs)
