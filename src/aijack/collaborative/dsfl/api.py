import copy

from ..core.api import BaseFLKnowledgeDistillationAPI


class DSFLAPI(BaseFLKnowledgeDistillationAPI):
    """API of DS-FL

    Args:
        server (DSFLServer): an instance of DSFLServer
        clients (List[DSFLClient]): a list of instances of DSFLClient
        public_dataloader (torch.DataLoader): a dataloader of public dataset
        local_dataloaders (List[torch.DataLoader]): a list of dataloaders of private dataests
        validation_dataloader (torch.DataLoader): a dataloader of validation dataset
        criterion (function): a loss function
        num_communication (int): number of communication
        device (str): device type
        server_optimizer (torch.Optimizer): a optimizer for the global model
        client_optimizers ([torch.Optimizer]): a list of optimizers for the local models
        epoch_local_training (int, optional): number of epochs of local training. Defaults to 1.
        epoch_global_distillation (int, optional): number of epochs of global distillation. Defaults to 1.
        epoch_local_distillation (int, optional): number of epochs of local distillation. Defaults to 1.
    """

    def __init__(
        self,
        server,
        clients,
        public_dataloader,
        local_dataloaders,
        criterion,
        num_communication,
        device,
        server_optimizer,
        client_optimizers,
        validation_dataloader=None,
        epoch_local_training=1,
        epoch_global_distillation=1,
        epoch_local_distillation=1,
        custom_action=lambda x: x,
    ):
        """Init DSFLAPI"""
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

        self.custom_action = custom_action
        self.epoch = 0

    def run(self):
        logging = {
            "loss_local": [],
            "loss_client_consensus": [],
            "loss_server_consensus": [],
            "acc_local": [],
            "acc_val": [],
        }
        for i in range(1, self.num_communication + 1):

            self.epoch = i

            for _ in range(self.epoch_local_training):
                loss_local = self.train_client(public=False)
            logging["loss_local"].append(loss_local)

            self.server.action()

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

            acc_on_local_dataset = self.local_score()
            print(f"epoch={i} acc on local datasets: ", acc_on_local_dataset)
            logging["acc_local"].append(acc_on_local_dataset)

            # validation
            if self.validation_dataloader is not None:
                acc_val = self.score(self.validation_dataloader)
                print(f"epoch={i} acc on validation dataset: ", acc_val)
                logging["acc_val"].append(copy.deepcopy(acc_val))

            self.custom_action(self)

        return logging
