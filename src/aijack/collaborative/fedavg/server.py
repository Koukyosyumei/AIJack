import numpy as np
import torch

from ..core import BaseServer
from ..optimizer import AdamFLOptimizer, SGDFLOptimizer


class FedAvgServer(BaseServer):
    def __init__(
        self,
        clients,
        global_model,
        server_id=0,
        lr=0.1,
        optimizer_type="sgd",
        optimizer_kwargs={},
    ):
        super(FedAvgServer, self).__init__(clients, global_model, server_id=server_id)
        self.lr = lr
        self._setup_optimizer(optimizer_type, **optimizer_kwargs)
        self.distribtue()

    def _setup_optimizer(self, optimizer_type, **kwargs):
        if optimizer_type == "sgd":
            self.optimizer = SGDFLOptimizer(
                self.server_model.parameters(), lr=self.lr, **kwargs
            )
        elif optimizer_type == "adam":
            self.optimizer = AdamFLOptimizer(
                self.server_model.parameters(), lr=self.lr, **kwargs
            )
        elif optimizer_type == "none":
            self.optimizer = None
        else:
            raise NotImplementedError(
                f"{optimizer_type} is not supported. You can specify `sgd`, `adam`, or `none`."
            )

    def action(self, use_gradients=True):
        self.receive(use_gradients)
        self.update(use_gradients)
        self.distribtue()

    def receive(self, use_gradients=True):
        if use_gradients:
            self.receive_local_gradients()
        else:
            self.receive_local_parameters()

    def update(self, use_gradients=True):
        if use_gradients:
            self.updata_from_gradients()
        else:
            self.update_from_parameters()

    def receive_local_gradients(self):
        self.uploaded_gradients = [c.upload_gradients() for c in self.clients]

    def receive_local_parameters(self):
        self.uploaded_parameters = [c.upload_parameters() for c in self.clients]

    def updata_from_gradients(self, weight=None):
        if weight is None:
            weight = np.ones(self.num_clients) / self.num_clients

        aggregated_gradients = [
            torch.zeros_like(params) for params in self.server_model.parameters()
        ]
        len_gradients = len(aggregated_gradients)

        for i, gradients in enumerate(self.uploaded_gradients):
            for gradient_id in range(len_gradients):
                aggregated_gradients[gradient_id] += weight[i] * gradients[gradient_id]

        self.optimizer.step(aggregated_gradients)
        # for params, grads in zip(self.server_model.parameters(), aggregated_gradients):
        #    params.data -= self.lr * grads

    def update_from_parameters(self, weight=None):
        if weight is None:
            weight = np.ones(self.num_clients) / self.num_clients

        averaged_params = self.uploaded_parameters[0]

        for k in averaged_params.keys():
            for i in range(0, len(self.uploaded_parameters)):
                local_model_params = self.uploaded_parameters[i]
                w = weight[i]
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        self.server_model.load_state_dict(averaged_params)

    def distribtue(self):
        for client in self.clients:
            client.download(self.server_model.state_dict())
