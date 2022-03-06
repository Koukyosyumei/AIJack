import copy

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from ..core.api import BaseFLKnowledgeDistillationAPI


class FedGEMSAPI(BaseFLKnowledgeDistillationAPI):
    def __init__(
        self,
        server,
        clients,
        public_dataloader,
        local_dataloaders,
        validation_dataloader,
        criterion,
        server_optimizer,
        client_optimizers,
        num_communication=10,
        epoch_client_on_localdataset=10,
        epoch_client_on_publicdataset=10,
        epoch_server_on_publicdataset=10,
        device="cpu",
    ):
        super().__init__(
            self,
            server,
            clients,
            public_dataloader,
            local_dataloaders,
            validation_dataloader,
            criterion,
            num_communication=num_communication,
            device=device,
        )
        self.server_optimizer = server_optimizer
        self.client_optimizers = client_optimizers

        self.epoch_client_on_localdataset = epoch_client_on_localdataset
        self.epoch_client_on_publicdataset = epoch_client_on_publicdataset
        self.epoch_server_on_publicdataset = epoch_server_on_publicdataset

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

        return loss_on_local_dataest

    def train_client_on_public_dataset(self):
        loss_on_public_dataset = []
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

            loss_on_public_dataset.append(
                copy.deepcopy(running_loss / len(self.public_dataloader))
            )

        return loss_on_public_dataset

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

    def run(self):
        logging = {
            "loss_client_local_dataset": [],
            "loss_server_public_dataset": [],
            "loss_client_public_dataset": [],
            "acc": [],
        }

        # train FedGEMS
        for epoch in range(1, self.num_communication + 1):
            for _ in range(self.epoch_client_on_localdataset):
                loss_client_local_dataset = self.train_client_on_local_dataset()
            for _ in range(self.epoch_server_on_publicdataset):
                loss_server_public_dataset = self.train_server_on_public_dataset()
            for _ in range(self.epoch_client_on_publicdataset):
                loss_client_public_dataset = self.train_client_on_public_dataset()

            print(
                f"epoch={epoch} loss_client_local_dataset: ", loss_client_local_dataset
            )
            logging["loss_client_local_dataset"].append(
                copy.deepcopy(loss_client_local_dataset)
            )
            print(
                f"epoch={epoch} loss_server_public_dataset: ",
                loss_server_public_dataset,
            )
            logging["loss_server_public_dataset"].append(
                copy.deepcopy(loss_server_public_dataset)
            )
            print(
                f"epoch={epoch} loss_client_public_dataset: ",
                loss_client_public_dataset,
            )
            logging["loss_client_public_dataset"].append(
                copy.deepcopy(loss_client_public_dataset)
            )

            if self.validation_dataloader is not None:
                acc = self.server_score(self.validation_dataloader)
                print(f"epoch={epoch} acc: ", acc)
                logging["acc"].append(copy.deepcopy(acc))

        return logging
