import copy

import numpy as np
import torch

from ..core import BaseServer
from ..core.utils import GRADIENTS_TAG, PARAMETERS_TAG
from ..optimizer import AdamFLOptimizer, SGDFLOptimizer


class FedAvgServer(BaseServer):
    """Server of FedAVG for single process simulation

    Args:
        clients ([FedAvgClient] | [int]): a list of FedAVG clients or their ids.
        global_model (torch.nn.Module): global model.
        server_id (int, optional): id of this server. Defaults to 0.
        lr (float, optional): learning rate. Defaults to 0.1.
        optimizer_type (str, optional): optimizer for the update of global model . Defaults to "sgd".
        server_side_update (bool, optional): If True, update the global model at the server-side. Defaults to True.
        optimizer_kwargs (dict, optional): kwargs for the global optimizer. Defaults to {}.
    """

    def __init__(
        self,
        clients,
        global_model,
        server_id=0,
        lr=0.1,
        optimizer_type="sgd",
        server_side_update=True,
        optimizer_kwargs={},
        device="cpu",
    ):
        super(FedAvgServer, self).__init__(clients, global_model, server_id=server_id)
        self.lr = lr
        self._setup_optimizer(optimizer_type, **optimizer_kwargs)
        self.server_side_update = server_side_update
        self.distribute(force_send_model_state_dict=True)
        self.device = device

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
        self.distribute()

    def receive(self, use_gradients=True):
        """Receive the local models

        Args:
            use_gradients (bool, optional): If True, receive the local gradients. Otherwise, receive the local parameters. Defaults to True.
        """
        if use_gradients:
            self.receive_local_gradients()
        else:
            self.receive_local_parameters()

    def update(self, use_gradients=True):
        """Update the global model

        Args:
            use_gradients (bool, optional): If True, update the global model with aggregated local gradients. Defaults to True.
        """
        if use_gradients:
            self.updata_from_gradients()
        else:
            self.update_from_parameters()

    def _preprocess_local_gradients(self, uploaded_grad):
        return uploaded_grad

    def receive_local_gradients(self):
        """Receive local gradients"""
        self.uploaded_gradients = [
            self._preprocess_local_gradients(c.upload_gradients()) for c in self.clients
        ]

    def receive_local_parameters(self):
        """Receive local parameters"""
        self.uploaded_parameters = [c.upload_parameters() for c in self.clients]

    def updata_from_gradients(self, weight=None):
        """Update the global model with the local gradients.

        Args:
            weight (list of float, optional): weight for each client. Defaults to None.
        """
        if weight is None:
            weight = np.ones(self.num_clients) / self.num_clients
            weight = weight.tolist()

        self.aggregated_gradients = [
            torch.zeros_like(params) for params in self.server_model.parameters()
        ]
        len_gradients = len(self.aggregated_gradients)

        for i, gradients in enumerate(self.uploaded_gradients):
            for gradient_id in range(len_gradients):
                self.aggregated_gradients[gradient_id] = (
                    gradients[gradient_id] * weight[i]
                    + self.aggregated_gradients[gradient_id]
                )

        if self.server_side_update:
            self.optimizer.step(self.aggregated_gradients)

    def update_from_parameters(self, weight=None):
        """Update the global model with the local model parameters.

        Args:
            weight (list of float, optional): weight for each client. Defaults to None.
        """
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

    def distribute(self, force_send_model_state_dict=False):
        """Distribute the current global model to each client.

        Args:
            force_send_model_state_dict (bool, optional): If True, send the global model as the dictionary of model state regardless of other parameters. Defaults to False.
        """
        for client in self.clients:
            if type(client) != int:
                if self.server_side_update or force_send_model_state_dict:
                    client.download(self.server_model.state_dict())
                else:
                    client.download(self.aggregated_gradients)


class MPIFedAvgServer:
    """MPI Wrapper for FedAvgServer

    Args:
        comm: MPI.COMM_WORLD
        server: the instance of FedAvgServer. The `clients` member variable shoud be the list of id.
    """

    def __init__(self, comm, server):
        self.comm = comm
        self.server = server
        self.num_clients = len(self.server.clients)

    def action(self):
        self.mpi_receive()
        self.server.update()
        self.mpi_distribute()
        self.round += 1

    def mpi_receive(self):
        self.mpi_receive_local_gradients()

    def mpi_receive_local_gradients(self):
        self.uploaded_gradients = []

        while len(self.uploaded_gradients) < self.num_clients:
            gradients_received = self.comm.recv(tag=GRADIENTS_TAG)
            self.uploaded_gradients.append(
                self.server._preprocess_local_gradients(gradients_received)
            )

    def mpi_distribute(self):
        # global_parameters = []
        # for params in self.server.server_model.parameters():
        #     global_parameters.append(copy.copy(params).reshape(-1).tolist())

        for client_id in self.server.clients:
            self.comm.send(
                list(self.server.server_model.parameters()),
                dest=client_id,
                tag=PARAMETERS_TAG,
            )

    def mpi_initialize(self):
        self.mpi_distribute()
