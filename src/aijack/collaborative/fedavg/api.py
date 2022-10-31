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
        self.clients_weight = [
            dataset_size / sum_local_dataset_sizes
            for dataset_size in local_dataset_sizes
        ]

    def local_train(self, com):
        for client_idx in range(self.client_num):
            client = self.clients[client_idx]
            trainloader = self.local_dataloaders[client_idx]
            optimizer = self.local_optimizers[client_idx]

            for i in range(self.local_epoch):
                running_loss = 0.0
                running_data_num = 0
                for _, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    inputs.requires_grad = True
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    client.zero_grad()

                    outputs = client(inputs)
                    loss = self.criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    running_data_num += inputs.shape[0]

                print(
                    f"communication {com}, epoch {i}: client-{client_idx+1}",
                    running_loss / running_data_num,
                )

    def run(self):
        for com in range(self.num_communication):
            self.local_train(com)
            self.server.receive(use_gradients=self.use_gradients)
            if self.use_gradients:
                self.server.updata_from_gradients(weight=self.clients_weight)
            else:
                self.server.update_from_parameters(weight=self.clients_weight)
            self.server.distribtue()

            self.custom_action(self)


class MPIFedAVGAPI(BaseFedAPI):
    def __init__(
        self,
        comm,
        mpiapi,
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
        self.mpiapi = mpiapi
        self.is_server = is_server
        self.criterion = criterion
        self.local_optimizer = local_optimizer
        self.local_dataloader = local_dataloader
        self.num_communication = num_communication
        self.local_epoch = local_epoch
        self.custom_action = custom_action
        self.device = device

    def run(self):
        self.mpiapi.mpi_initialize()
        self.comm.Barrier()

        for _ in range(self.num_communication):
            if not self.is_server:
                self.local_train()
            self.mpiapi.action()

            self.custom_action(self)
            self.comm.Barrier()

    def local_train(self):
        self.mpiapi.prev_parameters = []
        for param in self.mpiapi.client.model.parameters():
            self.mpiapi.client.prev_parameters.append(copy.deepcopy(param))

        for _ in range(self.local_epoch):
            running_loss = 0
            for (data, target) in self.local_dataloader:
                self.local_optimizer.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.mpiapi.client.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.local_optimizer.step()
                running_loss += loss.item()
