import torch


class Client(torch.nn.Module):
    def __init__(self, model, user_id=0):
        super().__init__()
        self.model = model
        self.user_id = user_id

    def forward(self, x):
        output = self.model(x)
        return output

    def upload(self):
        return self.model.state_dict()

    def download(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
