import copy

from ..core.api import BaseFedAPI


class FedAVGAPI(BaseFedAPI):
    """Implementation of FedAVG (McMahan, Brendan, et al. 'Communication-efficient learning of deep networks from decentralized data.' Artificial intelligence and statistics. PMLR, 2017.)

    Args:
        server (FedAvgServer): FedAVG server.
        clients ([FedAvgClient]): a list of FedAVG clients.
        criterion (function): loss function.
        local_optimizers ([torch.optimizer]): a list of local optimizers for clients
        local_dataloaders ([toch.dataloader]): a list of local dataloaders for clients
        num_communication (int, optional): number of communication. Defaults to 1.
        local_epoch (int, optional): number of epochs for local training within each communication. Defaults to 1.
        use_gradients (bool, optional): communicate gradients if True. Otherwise communicate parameters. Defaults to True.
        custom_action (function, optional): arbitrary function that takes this instance itself. Defaults to lambdax:x.
        device (str, optional): device type. Defaults to "cpu".
    """

    def __init__(
        self,
        server,
        clients,
        criterion,
        local_optimizers,
        local_dataloaders,
        num_communication=1,
        local_epoch=1,
        use_gradients=True,
        custom_action=lambda x: x,
        device="cpu",
    ):
        self.server = server
        self.clients = clients
        self.criterion = criterion
        self.local_optimizers = local_optimizers
        self.local_dataloaders = local_dataloaders
        self.num_communication = num_communication
        self.local_epoch = local_epoch
        self.use_gradients = use_gradients
        self.custom_action = custom_action
        self.device = device

        self.client_num = len(self.clients)

        local_dataset_sizes = [
            len(dataloader.dataset) for dataloader in self.local_dataloaders
        ]
        sum_local_dataset_sizes = sum(local_dataset_sizes)
        self.servre.weight = [
            dataset_size / sum_local_dataset_sizes
            for dataset_size in local_dataset_sizes
        ]

    def local_train(self, i):
        for client_idx in range(self.client_num):
            self.clients[client_idx].local_train(
                self.local_epoch,
                self.criterion,
                self.local_dataloaders[client_idx],
                self.local_optimizers[client_idx],
                communication_id=i,
            )

    def run(self):
        self.server.force_send_model_state_dict = True
        self.server.distribute()
        self.server.force_send_model_state_dict = False

        for i in range(self.num_communication):
            self.local_train(i)
            self.server.receive(use_gradients=self.use_gradients)
            if self.use_gradients:
                self.server.update_from_gradients()
            else:
                self.server.update_from_parameters()
            self.server.distribute()

            self.custom_action(self)


class MPIFedAVGAPI(BaseFedAPI):
    def __init__(
        self,
        comm,
        party,
        is_server,
        criterion,
        local_optimizer=None,
        local_dataloader=None,
        num_communication=1,
        local_epoch=1,
        custom_action=lambda x: x,
        device="cpu",
    ):
        self.comm = comm
        self.party = party
        self.is_server = is_server
        self.criterion = criterion
        self.local_optimizer = local_optimizer
        self.local_dataloader = local_dataloader
        self.num_communication = num_communication
        self.local_epoch = local_epoch
        self.custom_action = custom_action
        self.device = device

    def run(self):
        self.party.mpi_initialize()
        self.comm.Barrier()

        for i in range(self.num_communication):
            if not self.is_server:
                self.local_train(i)
            self.party.action()

            self.custom_action(self)
            self.comm.Barrier()

    def local_train(self, com_cnt):
        self.party.prev_parameters = []
        for param in self.party.model.parameters():
            self.party.prev_parameters.append(copy.deepcopy(param))

        self.party.local_train(
            self.local_epoch,
            self.criterion,
            self.local_dataloader,
            self.local_optimizer,
            communication_id=com_cnt,
        )
