import numpy as np
import torch
import torch.nn.functional as F

from ...manager import BaseManager

EPS = 1e-8


def calculate_cs(cs, num_clients, aggregate_historical_gradients):
    for i_idx in range(num_clients):
        for j_idx in range(i_idx + 1, num_clients):
            cs[i_idx][j_idx] = F.cosine_similarity(
                aggregate_historical_gradients[i_idx],
                aggregate_historical_gradients[j_idx],
                0,
                EPS,
            )
            cs[j_idx][i_idx] = cs[i_idx][j_idx]
    return cs


def normalize_cs(cs, v, num_clients):
    for i_idx in range(num_clients):
        for j_idx in range(num_clients):
            if v[j_idx] > v[i_idx]:
                cs[i_idx][j_idx] *= v[i_idx] / v[j_idx]
    return cs


def attach_foolsgold_to_server(cls):
    """Wraps the given class in FoolsGoldServerWrapper.

    Returns:
        cls: a class wrapped in FoolsGoldServerWrapper
    """

    class FoolsGoldServerWrapper(cls):
        """Implementation of https://arxiv.org/abs/1808.04866"""

        def __init__(self, *args, **kwargs):
            super(FoolsGoldServerWrapper, self).__init__(*args, **kwargs)

            tmp_flatten_local_gradient = torch.cat(
                [p.view(-1) for p in self.server_model.parameters()]
            ).to(self.device)
            self.aggregate_historical_gradients = [
                torch.zeros_like(tmp_flatten_local_gradient)
                for i in range(len(self.clients))
            ]
            self.cs = np.zeros((len(self.clients), len(self.clients)))
            self.v = np.zeros(len(self.clients))
            self.alpha = np.zeros(len(self.clients))

        def update(self):
            self.update_weight()
            self.update_from_gradients()

        def update_weight(self):
            """Updates weight for each client given the received local gradients."""
            for i, local_gradient in enumerate(self.uploaded_gradients):
                self.aggregate_historical_gradients[i] += torch.cat(
                    [g.to(self.device).view(-1) for g in local_gradient[1]]
                ).to(self.device)

            num_clients = len(self.uploaded_gradients)
            self.cs = self.calculate_cs(
                self.cs, num_clients, self.aggregate_historical_gradients
            )
            self.v = np.max(self.cs, axis=1)
            self.cs = self.normalize_cs(self.cs, self.v, num_clients)

            self.alpha = np.max(self.cs, axis=1)
            self.alpha = self.alpha / (np.max(self.alpha) + EPS)
            self.weight = self.alpha

    return FoolsGoldServerWrapper


class FoolsGoldServerManager(BaseManager):
    """Manager class for FoolsGold proposed in https://arxiv.org/abs/1808.04866."""

    def attach(self, cls):
        """Wraps the given class in FoolsGoldServerWrapper.

        Returns:
            cls: a class wrapped in FoolsGoldServerWrapper
        """
        return attach_foolsgold_to_server(cls, *self.args, **self.kwargs)
