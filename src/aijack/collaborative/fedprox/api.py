import copy

from ..fedavg import FedAVGAPI, MPIFedAVGAPI


class FedProxAPI(FedAVGAPI):
    """Implementation of FedProx (https://arxiv.org/abs/1812.06127)"""

    def __init__(self, *args, mu=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu

    def local_train(self, i):
        for client_idx in range(self.client_num):
            self.clients[client_idx].local_train(
                self.server.parameters(),
                self.local_epoch,
                self.criterion,
                self.local_dataloaders[client_idx],
                self.local_optimizers[client_idx],
                communication_id=i,
            )

    def run(self):
        for i in range(self.num_communication):
            self.local_train(i)

            self.server.receive(use_gradients=self.use_gradients)
            if self.use_gradients:
                self.server.updata_from_gradients(weight=self.clients_weight)
            else:
                self.server.update_from_parameters(weight=self.clients_weight)

            self.custom_action(self)


class MPIFedProxAPI(MPIFedAVGAPI):
    def __init__(self, *args, mu=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu

    def local_train(self, com_cnt):
        self.party.prev_parameters = []
        for param in self.party.model.parameters():
            self.party.prev_parameters.append(copy.deepcopy(param))

        self.party.client.local_train(
            self.party.prev_parameters,
            self.local_epoch,
            self.criterion,
            self.local_dataloader,
            self.local_optimizer,
            communication_id=com_cnt,
        )
