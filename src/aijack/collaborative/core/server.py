import torch


class BaseServer(torch.nn.Module):
    def __init__(self, clients, server_model, server_id=0):
        super().__init__()
        self.clients = clients
        self.server_id = server_id
        self.server_model = server_model
        self.num_clients = len(clients)

    def forward(self, x):
        return self.server_model(x)

    def update(self):
        pass

    def distribtue(self):
        pass

    def train(self):
        self.server_model.train()

    def eval(self):
        self.server_model.eval()
