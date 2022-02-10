import copy

from ..core import BaseClient


class FedAvgClient(BaseClient):
    def __init__(self, model, user_id=0, lr=0.1):
        super(FedAvgClient, self).__init__(model, user_id=user_id)
        self.lr = lr

        self.prev_parameters = []
        for param in self.model.parameters():
            self.prev_parameters.append(copy.deepcopy(param))

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
