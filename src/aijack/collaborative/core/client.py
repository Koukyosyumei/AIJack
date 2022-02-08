import torch


class BaseClient(torch.nn.Module):
    def __init__(self, model, user_id=0):
        super(BaseClient, self).__init__()
        self.model = model
        self.user_id = user_id

    def forward(self, x):
        return self.model(x)

    def upload(self):
        pass

    def download(self):
        pass

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
