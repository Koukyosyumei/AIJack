import torch


class Client(torch.nn.Module):
    def __init__(self, model, user_id=0):
        self.model = model
        self.user_id = user_id

    def forward(self, x):
        output = self.model(x)
        return output

    def upload(self, x):
        return self.forward(x)

    def download(self, x):
        pass

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
