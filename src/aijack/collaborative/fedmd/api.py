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
        num_communication=1,
        device="cpu",
        consensus_epoch=1,
        revisit_epoch=1,
        transfer_epoch_public=1,
        transfer_epoch_private=1,
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
        self.transfer_epoch_public = transfer_epoch_public
        self.transfer_epoch_private = transfer_epoch_private

    def run(self):
        logging = {
            "loss_client_local_dataset_transfer": [],
            "loss_client_public_dataset_transfer": [],
            "loss_client_consensus": [],
            "loss_client_revisit": [],
            "loss_server_public_dataset": [],
            "acc": [],
        }

        cnt = 0
        while True:
            loss_public = self.train_client(public=True)
            loss_local = self.train_client(public=False)
            print(f"epoch {cnt} (public - pretrain): {loss_local}")
            print(f"epoch {cnt} (local - pretrain): {loss_public}")
            logging["loss_client_public_dataset_transfer"].append(loss_public)
            logging["loss_client_local_dataset_transfer"].append(loss_local)

            cnt += 1
            if cnt >= self.transfer_epoch_public and cnt >= self.transfer_epoch_private:
                break

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
            if self.validation_dataloader is not None:
                acc = self.score(self.validation_dataloader)
                print(f"epoch={i} acc: ", acc)
                logging["acc"].append(copy.deepcopy(acc))

        return logging
