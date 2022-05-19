import copy

import torch

from ..core.api import BaseFLKnowledgeDistillationAPI


class FedGEMSAPI(BaseFLKnowledgeDistillationAPI):
    """API of FedGEMSAPI.

    Args:
        server (FedGEMSServer): a server.
        clients (List[FedGEMSClient]): a list of clients.
        public_dataloader (torch.utils.data.DataLoader): a dataloader of the public dataset.
        local_dataloaders (List[torch.utils.data.DataLoader]): a list of dataloaders of
                                                               the local datasets.
        validation_dataloader (torch.utils.data.DataLoader): a dataloader of the validation dataset.
        criterion (function)): a loss function
        server_optimizer (torch.optim.Optimizer): an optimizer for the global model
        client_optimizers (List[torch.optim.Optimizer]): a list of optimizers for the local models
        num_communication (int, optional): the number of communications. Defaults to 10.
        epoch_client_on_localdataset (int, optional): the number of epochs of client-side
                                                      training on the private datasets.
                                                      Defaults to 10.
        epoch_client_on_publicdataset (int, optional): the number of epochs of client-side
                                                       training on the public datasets.
                                                       Defaults to 10.
        epoch_server_on_publicdataset (int, optional): the number of epochs of server-side training
                                                       on the public dataset. Defaults to 10.
        device (str, optional): device type. Defaults to "cpu".
        custom_action (function, optional): custom function which this api calls at
                                            the end of every communication.
                                            Defaults to lambda x:x.
    """

    def __init__(
        self,
        server,
        clients,
        public_dataloader,
        local_dataloaders,
        criterion,
        server_optimizer,
        client_optimizers,
        validation_dataloader=None,
        num_communication=10,
        epoch_client_on_localdataset=10,
        epoch_client_on_publicdataset=10,
        epoch_server_on_publicdataset=10,
        device="cpu",
        custom_action=lambda x: x,
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

        self.custom_action = custom_action
        self.epoch = 0

    def train_client_on_public_dataset(self):
        """Train clients on the public dataset.

        Returns:
            List[float]: a list of average loss of each client.
        """
        loss_on_public_dataset = []
        for client_idx in range(self.client_num):
            client = self.clients[client_idx]
            optimizer = self.client_optimizers[client_idx]

            running_loss = 0.0
            for data in self.public_dataloader:
                idx, inputs, labels = data
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
        """Train the global model on the public dataset.

        Returns:
            float: average loss
        """
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

            self.epoch = epoch

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

            self.custom_action(self)

        return logging
