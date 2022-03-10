import copy

from ..core.api import BaseFLKnowledgeDistillationAPI


class DSFLAPI(BaseFLKnowledgeDistillationAPI):
    """Implementation of `Distillation-Based Semi-Supervised Federated Learning
    for Communication-Efficient Collaborative Training
    with Non-IID Private Data`"""

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
        server_optimizer,
        client_optimizers,
        epoch_local_training=1,
        epoch_global_distillation=1,
        epoch_local_distillation=1,
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
        self.server_optimizer = server_optimizer
        self.client_optimizers = client_optimizers
        self.epoch_local_training = epoch_local_training
        self.epoch_global_distillation = epoch_global_distillation
        self.epoch_local_distillation = epoch_local_distillation

    def run(self):
        logging = {
            "loss_local": [],
            "loss_client_consensus": [],
            "loss_server_consensus": [],
            "acc": [],
        }
        for i in range(1, self.num_communication + 1):
            for _ in range(self.epoch_local_training):
                loss_local = self.train_client(public=False)
            logging["loss_local"].append(loss_local)

            self.server.update()
            self.server.distribute()

            # distillation
            temp_consensus_loss = []
            for j, client in enumerate(self.clients):
                for _ in range(self.epoch_local_distillation):
                    consensus_loss = client.approach_consensus(
                        self.client_optimizers[j]
                    )
                temp_consensus_loss.append(consensus_loss)
            logging["loss_client_consensus"].append(temp_consensus_loss)

            for _ in range(self.epoch_global_distillation):
                loss_global = self.server.update_globalmodel(self.server_optimizer)
            logging["loss_server_consensus"].append(loss_global)

            print(f"epoch {i}: loss_local", loss_local)
            print(f"epoch {i}: loss_client_consensus", temp_consensus_loss)
            print(f"epoch {i}: loss_server_consensus", loss_global)

            # validation
            if self.validation_dataloader is not None:
                acc = self.score(self.validation_dataloader)
                print(f"epoch={i} acc: ", acc)
                logging["acc"].append(copy.deepcopy(acc))
