import numpy as np
import torch
import torch.nn.functional as F

from ...manager import BaseManager

EPS = 1e-8


def attach_foolsgold_to_server(cls):
    class FoolsGoldServerWrapper(cls):
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
            self.update_from_gradients(self.alpha)

        def update_weight(self):
            for i, local_gradient in enumerate(self.uploaded_gradients):
                self.aggregate_historical_gradients[i] += torch.cat(
                    [g.to(self.device).view(-1) for g in local_gradient[1]]
                ).to(self.device)

            num_clients = len(self.uploaded_gradients)

            for i_idx in range(num_clients):
                for j_idx in range(i + 1, num_clients):
                    self.cs[i_idx][j_idx] = F.cosine_similarity(
                        self.aggregate_historical_gradients[i_idx],
                        self.aggregate_historical_gradients[j_idx],
                        0,
                        EPS,
                    )
            self.v = np.max(self.cs, axis=1)

            for i_idx in range(num_clients):
                for j_idx in range(num_clients):
                    if i_idx == j_idx:
                        continue
                    if self.v[j_idx] > self.v[i_idx]:
                        self.cs[i_idx][j_idx] *= self.v[i_idx] / self.v[j_idx]

            self.alpha = np.max(self.cs, axis=1)
            self.alpha = self.alpha / (np.max(self.alpha) + EPS)

    return FoolsGoldServerWrapper


class FoolsGoldServerManager(BaseManager):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def attach(self, cls):
        return attach_foolsgold_to_server(cls, *self.args, **self.kwargs)
