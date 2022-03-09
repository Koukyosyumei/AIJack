import copy
from abc import abstractmethod

import numpy as np
import torch
from sklearn.metrics import accuracy_score


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

    def server_score(self, dataloader):
        in_preds = []
        in_label = []
        with torch.no_grad():
            for data in dataloader:
                _, inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).to(torch.int64)
                outputs = self.server(inputs)
                in_preds.append(outputs)
                in_label.append(labels)
            in_preds = torch.cat(in_preds)
            in_label = torch.cat(in_label)

        return accuracy_score(
            np.array(torch.argmax(in_preds, axis=1).cpu()), np.array(in_label.cpu())
        )
