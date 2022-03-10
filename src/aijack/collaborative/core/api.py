import copy
from abc import abstractmethod

import torch

from ...utils import accuracy_torch_dataloader


class BaseFLKnowledgeDistillationAPI:
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
        server_score = accuracy_torch_dataloader(
            self.server, dataloader, cpu=self.device
        )
        clients_score = [
            accuracy_torch_dataloader(client, dataloader, cpu=self.device)
            for client in self.clients
        ]
        return {"server_score": server_score, "clients_score": clients_score}
