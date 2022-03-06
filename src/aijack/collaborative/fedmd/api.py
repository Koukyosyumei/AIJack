import copy

from ..core.api import BaseFLKnowledgeDistillationAPI


class FedMDAPI(BaseFLKnowledgeDistillationAPI):
    def __init__(
        self,
        server,
        clients,
        public_dataloader,
        local_dataloaders,
        validation_dataloader,
        criterion,
        client_optimizers,
        num_communication=10,
        device="cpu",
        consensus_epoch=1,
        revisit_epoch=1,
        transfer_epoch=10,
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
        self.consensus_epoch = consensus_epoch
        self.revisit_epoch = revisit_epoch
        self.transfer_epoch = transfer_epoch

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
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                loss = self.criterion(client(x), y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            loss_on_local_dataest.append(copy.deepcopy(running_loss / len(trainloader)))

        return loss_on_local_dataest

    def run(self):
        logging = {
            "loss_client_local_dataset_transfer": [],
            "loss_client_public_dataset_transfer": [],
            "loss_client_consensus": [],
            "loss_client_revisit": [],
            "loss_server_public_dataset": [],
            "acc": [],
        }

        for i in range(self.transfer_epoch):
            loss_public = self.train_client(public=True)
            loss_local = self.train_client(public=False)
            print(f"epoch {i} (public - pretrain): {loss_local}")
            print(f"epoch {i} (local - pretrain): {loss_public}")
            logging["loss_client_public_dataset_transfer"].append(loss_public)
            logging["loss_client_local_dataset_transfer"].append(loss_local)

        for i in range(1, self.num_communication + 1):
            self.server.update()
            self.server.distribute()

            # Digest
            temp_consensus_loss = []
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

            # evaluation
            temp_acc_list = []
            for j, client in enumerate(self.clients):
                acc = client.score(self.validation_dataloader)
                print(f"client {j} acc score is ", acc)
                temp_acc_list.append(acc)
            logging["acc"].append(temp_acc_list)

        return logging
