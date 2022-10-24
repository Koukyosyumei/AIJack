import copy


class FedAVGAPI:
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

    def run(self):
        for com in range(self.num_communication):
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

            self.server.receive(use_gradients=self.use_gradients)
            if self.use_gradients:
                self.server.updata_from_gradients(weight=self.clients_weight)
            else:
                self.server.update_from_parameters(weight=self.clients_weight)
            self.server.distribtue()

            self.custom_action(self)


class MPIFedAVGAPI:
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
        if self.is_server:
            self.party.send_parameters()
        else:
            self.party.download()
        self.comm.Barrier()

        for _ in range(self.num_communication):
            if self.is_server:
                self.party.action()
            else:
                self.local_train()
                self.party.upload()
                self.party.model.zero_grad()
                self.party.download()

            self.custom_action(self)
            self.comm.Barrier()

    def local_train(self):
        self.party.prev_parameters = []
        for param in self.party.model.parameters():
            self.party.prev_parameters.append(copy.deepcopy(param))

        for _ in range(self.local_epoch):
            running_loss = 0
            for (data, target) in self.local_dataloader:
                self.local_optimizer.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.party.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.local_optimizer.step()
                running_loss += loss.item()
