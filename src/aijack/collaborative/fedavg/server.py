import numpy as np
import torch

from ..core import BaseServer


class FedAvgServer(BaseServer):
    def __init__(self, clients, global_model, server_id=0, lr=0.1):
        super(FedAvgServer, self).__init__(clients, global_model, server_id=server_id)
        self.lr = lr
        self.distribtue()

    def action(self, gradients=True):
        self.update(gradients)
        self.distribtue()

    def update(self, gradients=True):
        if gradients:
            self.updata_from_gradients()
        else:
            self.update_from_parameters()

    def updata_from_gradients(self, weight=None):
        if weight is None:
            weight = np.ones(self.num_clients) / self.num_clients

        self.uploaded_gradients = [c.upload_gradients() for c in self.clients]
        aggregated_gradients = [
            torch.zeros_like(params) for params in self.server_model.parameters()
        ]
        len_gradients = len(aggregated_gradients)

        for gradients in self.uploaded_gradients:
            for gradient_id in range(len_gradients):
                aggregated_gradients[gradient_id] += (1 / self.num_clients) * gradients[
                    gradient_id
                ]

        for params, grads in zip(self.server_model.parameters(), aggregated_gradients):
            params.data -= self.lr * grads

    def update_from_parameters(self, weight=None):
        if weight is None:
            weight = np.ones(self.num_clients) / self.num_clients

        uploaded_parameters = [c.upload_parameters() for c in self.clients]
        averaged_params = uploaded_parameters[0]

        for k in averaged_params.keys():
            for i in range(0, len(uploaded_parameters)):
                local_model_params = uploaded_parameters[i]
                w = weight[i]
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        self.server_model.load_state_dict(averaged_params)

    def distribtue(self):
        for client in self.clients:
            client.download(self.server_model.state_dict())
