import torch

from ...utils.metrics import crossentropyloss_between_logits
from ...utils.utils import torch_round_x_decimal
from ..core import BaseClient


class DSFLClient(BaseClient):
    """Client of DS-FL.

    Args:
        model (torch.nn.Module): _description_
        public_dataloader (torch.utils.data.DataLoader): a dataloader of the public dataset.
        output_dim (int, optional): the dimension of the output. Defaults to 1.
        round_decimal (int, optional): number of digits to round up. Defaults to None.
        device (str, optional): device type. Defaults to "cpu".
        user_id (int, optional): id of this client. Defaults to 0.
    """

    def __init__(
        self,
        model,
        public_dataloader,
        output_dim=1,
        round_decimal=None,
        device="cpu",
        user_id=0,
    ):
        """Init DSFLClient."""
        super().__init__(model, user_id)
        self.public_dataloader = public_dataloader
        self.round_decimal = round_decimal
        self.device = device
        self.global_logit = None

        len_public_dataloader = len(self.public_dataloader.dataset)
        self.logit2server = torch.ones((len_public_dataloader, output_dim)).to(
            self.device
        ) * float("inf")

    def upload(self):
        """Upload the output logits on the public dataset to the server.

        Returns:
            torch.Tensor: the output logits of the public dataset.
        """
        for data in self.public_dataloader:
            idx = data[0]
            x = data[1]
            x = x.to(self.device)
            self.logit2server[idx, :] = self(x).detach().softmax(dim=-1)

        if self.round_decimal is None:
            return self.logit2server
        else:
            return torch_round_x_decimal(self.logit2server, self.round_decimal)

    def download(self, global_logit):
        """Download the global logits from the server.

        Args:
            global_logit (torch.Tensor): the global logits from the server
        """
        self.global_logit = global_logit

    def approach_consensus(self, consensus_optimizer):
        """Train the own local model to minimize the distance between the global logits and
        the output logits of the local model on the public dataset.

        Args:
            consensus_optimizer (torch.optim.Optimizer): an optimizer to train the local model.

        Returns:
            float: averaged loss.
        """
        running_loss = 0
        for global_data in self.public_dataloader:
            idx = global_data[0]
            x = global_data[1].to(self.device)
            y_global = self.global_logit[idx, :].to(self.device).detach()
            consensus_optimizer.zero_grad()
            y_local = self(x)
            loss_consensus = crossentropyloss_between_logits(y_local, y_global)
            loss_consensus.backward()
            consensus_optimizer.step()
            running_loss += loss_consensus.item()
        running_loss /= len(self.public_dataloader)
        return running_loss
