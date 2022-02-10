import copy

from ..core import BaseClient


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
