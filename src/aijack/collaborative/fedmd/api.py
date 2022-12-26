import copy

import torch

from ..core.api import BaseFedAPI, BaseFLKnowledgeDistillationAPI


class FedMDAPI(BaseFLKnowledgeDistillationAPI):
    """Implementation of `Fedmd: Heterogenous federated learning via model distillation`"""

    def __init__(
        self,
        server,
        clients,
        public_dataloader,
        local_dataloaders,
        criterion,
        client_optimizers,
        validation_dataloader=None,
        server_optimizer=None,
        num_communication=1,
        device="cpu",
        consensus_epoch=1,
        revisit_epoch=1,
        transfer_epoch_public=1,
        transfer_epoch_private=1,
        server_training_epoch=1,
        custom_action=lambda x: x,
    ):
        super().__init__(
            server,
            clients,
            public_dataloader,
            local_dataloaders,
            validation_dataloader,
            criterion,
            num_communication,
            device,
        )
        self.client_optimizers = client_optimizers
        self.server_optimizer = server_optimizer
        self.consensus_epoch = consensus_epoch
        self.revisit_epoch = revisit_epoch
        self.transfer_epoch_public = transfer_epoch_public
        self.transfer_epoch_private = transfer_epoch_private
        self.server_training_epoch = server_training_epoch
        self.custom_action = custom_action
        self.epoch = 0

    def train_server(self):
        if self.server_optimizer is None:
            return 0.0

        running_loss = 0.0
        for data in self.public_dataloader:
            _, x, y = data
            x = x.to(self.device)
            y = y.to(self.device).to(torch.int64)

            self.server_optimizer.zero_grad()
            loss = self.criterion(self.server(x), y)
            loss.backward()
            self.server_optimizer.step()

            running_loss += loss.item()

        running_loss /= len(self.public_dataloader)

        return running_loss

    def run(self):
        logging = {
            "loss_client_local_dataset_transfer": [],
            "loss_client_public_dataset_transfer": [],
            "loss_client_consensus": [],
            "loss_client_revisit": [],
            "loss_server_public_dataset": [],
            "acc_local": [],
            "acc_pub": [],
            "acc_val": [],
        }

        # Transfer
        for i in range(1, self.transfer_epoch_public + 1):
            loss_public = self.train_client(public=True)
            print(f"epoch {i} (public - pretrain): {loss_public}")
            logging["loss_client_public_dataset_transfer"].append(loss_public)

        if self.validation_dataloader is not None:
            acc_val = self.score(
                self.validation_dataloader, self.server_optimizer is None
            )
            print("acc on validation dataset: ", acc_val)
            logging["acc_val"].append(copy.deepcopy(acc_val))

        for i in range(1, self.transfer_epoch_private + 1):
            loss_local = self.train_client(public=False)
            print(f"epoch {i} (local - pretrain): {loss_local}")
            logging["loss_client_local_dataset_transfer"].append(loss_local)

        if self.validation_dataloader is not None:
            acc_val = self.score(
                self.validation_dataloader, self.server_optimizer is None
            )
            print("acc on validation dataset: ", acc_val)
            logging["acc_val"].append(copy.deepcopy(acc_val))

        for i in range(1, self.num_communication + 1):

            self.epoch = i

            self.server.action()

            # Digest
            temp_consensus_loss = []
            if len(self.clients) > 1:
                for j, client in enumerate(self.clients):
                    for _ in range(self.consensus_epoch):
                        consensus_loss = client.approach_consensus(
                            self.client_optimizers[j]
                        )
                    print(f"epoch {i}, client {j}: {consensus_loss}")
                    temp_consensus_loss.append(consensus_loss)
            logging["loss_client_consensus"].append(temp_consensus_loss)

            # Revisit
            for _ in range(self.revisit_epoch):
                loss_local_revisit = self.train_client(public=False)
            logging["loss_client_revisit"].append(loss_local_revisit)

            # Train a server-side model if it exists (different from the original paper)
            for _ in range(self.server_training_epoch):
                loss_server_public = self.train_server()
            logging["loss_server_public_dataset"].append(loss_server_public)

            acc_on_local_dataset = self.local_score()
            print(f"epoch={i} acc on local datasets: ", acc_on_local_dataset)
            logging["acc_local"].append(acc_on_local_dataset)
            acc_pub = self.score(self.public_dataloader, self.server_optimizer is None)
            print(f"epoch={i} acc on public dataset: ", acc_pub)
            logging["acc_pub"].append(copy.deepcopy(acc_pub))
            # evaluation
            if self.validation_dataloader is not None:
                acc_val = self.score(
                    self.validation_dataloader, self.server_optimizer is None
                )
                print(f"epoch={i} acc on validation dataset: ", acc_val)
                logging["acc_val"].append(copy.deepcopy(acc_val))

            self.custom_action(self)

        return logging


class MPIFedMDAPI(BaseFedAPI):
    def __init__(
        self,
        comm,
        party,
        is_server,
        criterion,
        local_optimizer=None,
        local_dataloader=None,
        public_dataloader=None,
        num_communication=1,
        local_epoch=1,
        consensus_epoch=1,
        revisit_epoch=1,
        transfer_epoch_public=1,
        transfer_epoch_private=1,
        custom_action=lambda x: x,
        device="cpu",
    ):
        self.comm = comm
        self.party = party
        self.is_server = is_server
        self.criterion = criterion
        self.local_optimizer = local_optimizer
        self.local_dataloader = local_dataloader
        self.public_dataloader = public_dataloader
        self.num_communication = num_communication
        self.local_epoch = local_epoch
        self.consensus_epoch = consensus_epoch
        self.revisit_epoch = revisit_epoch
        self.transfer_epoch_public = transfer_epoch_public
        self.transfer_epoch_private = transfer_epoch_private
        self.custom_action = custom_action
        self.device = device

    def run(self):
        for _ in range(self.num_communication):

            # Transfer phase
            if not self.is_server:
                for _ in range(1, self.transfer_epoch_public + 1):
                    self.train_client(public=True)
                for _ in range(1, self.transfer_epoch_private + 1):
                    self.train_client(public=False)

            # Updata global logits
            for _ in range(1, self.num_communication + 1):
                self.party.action()
            self.comm.Barrier()

            # Digest phase
            if not self.is_server:
                for _ in range(1, self.consensus_epoch):
                    self.party.client.approach_consensus(self.local_optimizer)

            # Revisit phase
            for _ in range(self.revisit_epoch):
                self.train_client(public=False)

            self.party.action()

            self.custom_action(self)
            self.comm.Barrier()

    def train_client(self, public=True):
        for _ in range(self.local_epoch):
            running_loss = 0
            trainloader = self.public_dataloader if public else self.local_dataloader
            for (_, data, target) in trainloader:
                self.local_optimizer.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.party.client.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.local_optimizer.step()
                running_loss += loss.item()
