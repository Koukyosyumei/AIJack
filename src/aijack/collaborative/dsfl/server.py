import torch

from ...utils.metrics import crossentropyloss_between_logits
from ..core import BaseServer


class DSFLServer(BaseServer):
    """Server of DS-FL

    Args:
        clients (Llist[torch.nn.Module]): a list of clients.
        global_model (torch.nn.Module): the global model
        public_dataloader (torch.utils.data.DataLoader): a dataloader of the public dataset
        aggregation (str, optional): the type of the aggregation of the logits. Defaults to "ERA".
        distillation_loss_name (str, optional): the type of the loss function fot the distillation loss.
                                                Defaults to "crossentropy".
        era_temperature (float, optional): the temperature of ERA. Defaults to 0.1.
        server_id (int, optional): the id of this server. Defaults to 0.
        device (str, optional): device type. Defaults to "cpu".
    """

    def __init__(
        self,
        clients,
        global_model,
        public_dataloader,
        aggregation="ERA",
        distillation_loss_name="crossentropy",
        era_temperature=0.1,
        server_id=0,
        device="cpu",
    ):
        """Init DSFLServer"""
        super(DSFLServer, self).__init__(clients, global_model, server_id=server_id)
        self.public_dataloader = public_dataloader
        self.aggregation = aggregation
        self.era_temperature = era_temperature
        self.consensus = None
        self.device = device

        self._set_distillation_loss(distillation_loss_name)

    def _set_distillation_loss(self, name):
        """Setup the loss function for distillation.
        `crossentropy`, `L2` or `L1`.

        Args:
            name (str): type of the function

        Raises:
            NotImplementedError: Raises when `name` is not supported.
        """
        if name == "crossentropy":
            self.distillation_loss = crossentropyloss_between_logits
        elif name == "L2":
            self.distillation_loss = torch.nn.MSELoss()
        elif name == "L1":
            self.distillation_loss = torch.nn.L1Loss()
        else:
            raise NotImplementedError(f"{name} is not supported")

    def action(self):
        self.update()
        self.distribute()

    def update(self):
        """Update the aggregated consensus logits with the output logits received from the clients.

        Raises:
            NotImplementedError: Raises when the specified aggregation type is not supported.
        """
        if self.aggregation == "ERA":
            self._entropy_reduction_aggregation()
        elif self.aggregation == "SA":
            self._simple_aggregation()
        else:
            raise NotImplementedError(f"{self.aggregation} is not supported")

    def update_globalmodel(self, global_optimizer):
        """Train the global model with the global consensus logits.

        Args:
            global_optimizer (torch.optim.Optimizer): an optimizer

        Returns:
            float: average loss
        """
        running_loss = 0
        for global_data in self.public_dataloader:
            idx = global_data[0]
            x = global_data[1].to(self.device)
            y_global = self.consensus[idx, :].to(self.device)
            global_optimizer.zero_grad()
            y_pred = self(x)
            loss_consensus = self.distillation_loss(y_pred, y_global)
            loss_consensus.backward()
            global_optimizer.step()
            running_loss += loss_consensus.item()
        running_loss /= len(self.public_dataloader)
        return running_loss

    def distribute(self):
        """Distribute the logits of public dataset to each client."""
        for client in self.clients:
            client.download(self.consensus)

    def _entropy_reduction_aggregation(self):
        """Aggregate the received logits with ERA"""
        self._simple_aggregation()
        self.consensus = torch.softmax(self.consensus / self.era_temperature, dim=1)

    def _simple_aggregation(self):
        """Aggregate the received logits with SA (calculating average)"""
        self.consensus = self.clients[0].upload() / len(self.clients)
        for client in self.clients[1:]:
            self.consensus += client.upload() / len(self.clients)
