import copy

import numpy as np
import torch
from mpi4py import MPI

from ..core import BaseServer
from ..core.utils import GRADIENTS_TAG, PARAMETERS_TAG, RECEIVE_NAN_CODE
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


class MPIFedAVGServer(BaseServer):
    def __init__(
        self,
        comm,
        global_model,
        myid,
        client_ids,
        server_id=0,
        lr=0.1,
        optimizer_type="sgd",
        optimizer_kwargs={},
    ):
        super(MPIFedAVGServer, self).__init__(None, global_model, server_id=server_id)
        self.comm = comm
        self.myid = myid
        self.client_ids = client_ids

        self.lr = lr

        self.round = 1
        self.num_clients = len(client_ids)

        self._setup_optimizer(optimizer_type, **optimizer_kwargs)
        self.send_parameters()

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

    def send_parameters(self):
        global_parameters = []
        for params in self.model.parameters():
            global_parameters.append(copy.copy(params).reshape(-1).tolist())

        for client_id in self.client_ids:
            self.comm.send(global_parameters, dest=client_id, tag=PARAMETERS_TAG)

    def action(self):
        self.receive()
        self.updata()
        self.send_parameters()

    def receive(self):
        self.receive_local_gradients()

    def receive_local_gradients(self):
        self.received_gradients = []

        while len(self.received_gradients) < self.num_clients:
            gradients_flattend = self.comm.recv(tag=GRADIENTS_TAG)
            gradients_reshaped = []
            for params, grad in zip(self.model.parameters(), gradients_flattend):
                gradients_reshaped.append(
                    torch.Tensor(grad).to(self.device).reshape(params.shape)
                )
                if torch.sum(torch.isnan(gradients_reshaped[-1])):
                    print("the received gradients contains nan")
                    MPI.COMM_WORLD.Abort(RECEIVE_NAN_CODE)

            self.received_gradients.append(gradients_reshaped)

    def update(self):
        self.updata_from_gradients()

    def _aggregate(self):
        self.aggregated_gradients = [
            torch.zeros_like(params) for params in self.model.parameters()
        ]
        len_gradients = len(self.aggregated_gradients)

        for gradients in self.received_gradients:
            for gradient_id in range(len_gradients):
                self.aggregated_gradients[gradient_id] += (
                    1 / self.num_clients
                ) * gradients[gradient_id]

    def updata_from_gradients(self):
        self._aggregate()
        self.optimizer.step(self.aggregated_gradients)
