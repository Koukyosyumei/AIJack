import copy

import numpy as np
import torch
from sklearn.metrics import accuracy_score


class FedGEMSAPI:
    def __init__(
        self,
        server,
        clients,
        public_dataloader,
        local_dataloaders,
        server_optimizer,
        client_optimizers,
        criterion,
        device="cpu",
    ):
        self.server = server
        self.clients = clients
        self.public_dataloader = public_dataloader
        self.local_dataloaders = local_dataloaders
        self.server_optimizer = server_optimizer
        self.client_optimizers = client_optimizers
        self.criterion = criterion
        self.device = device

        self.client_num = len(clients)

    def train_client_on_local_dataset(self):
        loss_on_local_dataest = []
        for client_idx in range(self.client_num):
            client = self.clients[client_idx]
            trainloader = self.local_dataloaders[client_idx]
            optimizer = self.client_optimizers[client_idx]

            running_loss = 0.0
            for data in trainloader:
                _, x, y = data
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                loss = self.criterion(client(x), y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            loss_on_local_dataest.append(copy.deepcopy(running_loss / len(trainloader)))

    def train_client_on_public_dataset(self):
        for client_idx in range(self.client_num):
            client = self.clients[client_idx]
            optimizer = self.client_optimizers[client_idx]

            running_loss = 0.0
            for data in self.public_dataloader:
                idx, inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                y_preds = client(inputs)
                loss = client.culc_loss_on_public_dataset(idx, y_preds, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

    def train_server_on_public_dataset(self):
        server_running_loss = 0
        for data in self.public_dataloader:
            idx, x, y = data
            x = x.to(self.device)
            y = y.to(self.device)

            self.server_optimizer.zero_grad()
            server_loss = self.server.self_evaluation_on_the_public_dataset(idx, x, y)

            total_loss = server_loss  # + 5*attack_loss
            total_loss.backward()
            self.server_optimizer.step()
            self.server.update(idx, x)

            server_running_loss += server_loss.item() / len(self.public_dataloader)

        self.server.action()
        return server_running_loss

    def server_score(self, dataloader):
        in_preds = []
        in_label = []
        with torch.no_grad():
            for data in dataloader:
                _, inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.server(inputs)
                in_preds.append(outputs)
                in_label.append(labels)
            in_preds = torch.cat(in_preds)
            in_label = torch.cat(in_label)

        return accuracy_score(
            np.array(torch.argmax(in_preds, axis=1).cpu()), np.array(in_label.cpu())
        )
