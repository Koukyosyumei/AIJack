import copy

import torch

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

    def train_client_on_public_dataset(self):
        loss_on_public_dataset = []
        for client_idx in range(self.client_num):
            client = self.clients[client_idx]
            optimizer = self.client_optimizers[client_idx]

            running_loss = 0.0
            for data in self.public_dataloader:
                idx, inputs, labels = data
                print(idx)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).to(torch.int64)

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
            y = y.to(self.device).to(torch.int64)

            self.server_optimizer.zero_grad()
            server_loss = self.server.self_evaluation_on_the_public_dataset(idx, x, y)

            total_loss = server_loss  # + 5*attack_loss
            total_loss.backward()
            self.server_optimizer.step()
            self.server.update(idx, x)

            server_running_loss += server_loss.item() / len(self.public_dataloader)

        self.server.action()
        return server_running_loss

    def run(self):
        logging = {
            "loss_client_local_dataset": [],
            "loss_server_public_dataset": [],
            "loss_client_public_dataset": [],
            "acc_local": [],
            "acc_pub": [],
            "acc_val": [],
        }

        # train FedGEMS
        for epoch in range(1, self.num_communication + 1):
            for _ in range(self.epoch_client_on_localdataset):
                loss_client_local_dataset = self.train_client(public=False)
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

            acc_on_local_dataset = self.local_score()
            print(f"epoch={epoch} acc on local datasets: ", acc_on_local_dataset)
            logging["acc_local"].append(acc_on_local_dataset)
            acc_pub = self.score(self.public_dataloader)
            print(f"epoch={epoch} acc on public dataset: ", acc_pub)
            logging["acc_pub"].append(copy.deepcopy(acc_pub))
            # evaluation
            if self.validation_dataloader is not None:
                acc_val = self.score(self.validation_dataloader)
                print(f"epoch={epoch} acc on validation dataset: ", acc_val)
                logging["acc_val"].append(copy.deepcopy(acc_val))

        return logging
