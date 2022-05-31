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
            dataloader.dataset.x.shape[0] for dataloader in self.local_dataloaders
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
                optimizer = self.client_optimizers[client_idx]

                for i in range(self.local_epoch):
                    running_loss = 0.0
                    running_data_num = 0
                    for _, data in enumerate(trainloader, 0):
                        _, inputs, labels = data
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
