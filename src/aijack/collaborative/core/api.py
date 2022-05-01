import copy
from abc import abstractmethod

import torch

from ...utils import accuracy_torch_dataloader


class BaseFLKnowledgeDistillationAPI:
    """Abstract class for API of federated learning with knowledge distillation.

    Args:
        server (aijack.collaborative.core.BaseServer): the server
        clients (List[aijack.collaborative.core.BaseClient]): a list of the clients
        public_dataloader (torch.utils.data.DataLoader): a dataloader for the public dataset
        local_dataloaders (List[torch.utils.data.DataLoader]): a list of local dataloaders
        validation_dataloader (torch.utils.data.DataLoader): a dataloader for the validation dataset
        criterion (function): a function to calculate the loss
        num_communication (int): the number of communication
        device (str): device type
    """

    def __init__(
        self,
        server,
        clients,
        public_dataloader,
        local_dataloaders,
        validation_dataloader,
        criterion,
        num_communication,
        device,
    ):
        """Initialize BaseFLKnowledgeDistillationAPI"""
        self.server = server
        self.clients = clients
        self.public_dataloader = public_dataloader
        self.local_dataloaders = local_dataloaders
        self.validation_dataloader = validation_dataloader
        self.criterion = criterion
        self.num_communication = num_communication
        self.device = device

        self.client_num = len(clients)

    def train_client(self, public=True):
        """Train local models with the local datasets or the public dataset.

        Args:
            public (bool, optional): Train with the public dataset or the local datasets.
                                     Defaults to True.

        Returns:
            List[float]: a list of average loss of each clients.
        """
        loss_on_local_dataest = []
        for client_idx in range(self.client_num):
            client = self.clients[client_idx]
            if public:
                trainloader = self.public_dataloader
            else:
                trainloader = self.local_dataloaders[client_idx]
            optimizer = self.client_optimizers[client_idx]

            running_loss = 0.0
            for data in trainloader:
                _, x, y = data
                x = x.to(self.device)
                y = y.to(self.device).to(torch.int64)

                optimizer.zero_grad()
                loss = self.criterion(client(x), y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            loss_on_local_dataest.append(copy.deepcopy(running_loss / len(trainloader)))

        return loss_on_local_dataest

    @abstractmethod
    def run(self):
        pass

    def score(self, dataloader):
        """Returns the performance on the given dataset.

        Args:
            dataloader (torch.utils.data.DataLoader): a dataloader

        Returns:
            Dict[str, int]: performance of global model and local models
        """
        server_score = accuracy_torch_dataloader(
            self.server, dataloader, device=self.device
        )
        clients_score = [
            accuracy_torch_dataloader(client, dataloader, device=self.device)
            for client in self.clients
        ]
        return {"server_score": server_score, "clients_score": clients_score}

    def local_score(self):
        """Returns the local performance of each clients.

        Returns:
            Dict[str, int]: performance of global model and local models
        """
        local_score_list = []
        for client, local_dataloader in zip(self.clients, self.local_dataloaders):
            temp_score = accuracy_torch_dataloader(
                client, local_dataloader, device=self.device
            )
            local_score_list.append(temp_score)

        return {"clients_score": local_score_list}
