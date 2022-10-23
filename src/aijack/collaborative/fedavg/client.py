import copy

import torch

from ..core import BaseClient
from ..core.utils import GRADIENTS_TAG, PARAMETERS_TAG


class FedAvgClient(BaseClient):
    def __init__(self, model, user_id=0, lr=0.1, send_gradient=True):
        super(FedAvgClient, self).__init__(model, user_id=user_id)
        self.lr = lr
        self.send_gradient = send_gradient

        self.prev_parameters = []
        for param in self.model.parameters():
            self.prev_parameters.append(copy.deepcopy(param))

    def upload(self):
        if self.send_gradient:
            return self.upload_gradients()
        else:
            return self.upload_parameters()

    def upload_parameters(self):
        return self.model.state_dict()

    def upload_gradients(self):
        gradients = []
        for param, prev_param in zip(self.model.parameters(), self.prev_parameters):
            gradients.append((prev_param - param) / self.lr)
        return gradients

    def download(self, model_parameters):
        self.model.load_state_dict(model_parameters)

        self.prev_parameters = []
        for param in self.model.parameters():
            self.prev_parameters.append(copy.deepcopy(param))


class MPIFedAVGClient(BaseClient):
    def __init__(self, comm, model, user_id=0, lr=0.1, device="cpu"):
        super(MPIFedAVGClient, self).__init__(model, user_id=user_id)
        self.comm = comm
        self.lr = lr
        self.device = device

        self.prev_parameters = []
        for param in self.model.parameters():
            self.prev_parameters.append(copy.deepcopy(param))

    def upload(self):
        self.upload_gradient()

    def upload_gradient(self, destination_id=0):
        self.gradients = []
        for param, prev_param in zip(self.model.parameters(), self.prev_parameters):
            self.gradients.append((prev_param.reshape(-1) - param.reshape(-1)).tolist())
        self.comm.send(self.gradients, dest=destination_id, tag=GRADIENTS_TAG)

    def download(self):
        new_parameters = self.comm.recv(tag=PARAMETERS_TAG)
        for params, new_params in zip(self.model.parameters(), new_parameters):
            params.data = torch.Tensor(new_params).reshape(params.shape).to(self.device)
