import torch


class BaseClient(torch.nn.Module):
    """Abstract class foe the client of collaborative learning.

    Args:
        model (torch.nn.Module): a local model
        user_id (int, optional): id of this client. Defaults to 0.
    """

    def __init__(self, model, user_id=0):
        """Initialize BaseClient"""
        super(BaseClient, self).__init__()
        self.model = model
        self.user_id = user_id

    def forward(self, x):
        return self.model(x)

    def upload(self):
        """Upload the locally learned informatino to the server."""
        pass

    def download(self):
        """Download the global model from the server."""
        pass

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def backward(self, loss):
        """Execute backward mode automatic differentiation with the give loss.

        Args:
            loss (torch.Tensor): the value of calculated loss.
        """
        loss.backward()
