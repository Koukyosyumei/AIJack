import copy


class FedMDAPI:
    def __init__(
        self,
        server,
        clients,
        public_dataloader,
        local_dataloaders,
        client_optimizers,
        criterion,
        pretrain_epoch=10,
        transfer_epoch=10,
        device="cpu",
    ):
        self.server = server
        self.clients = clients
        self.public_dataloader = public_dataloader
        self.local_dataloaders = local_dataloaders
        self.client_optimizers = client_optimizers
        self.criterion = criterion
        self.pretrain_epoch = pretrain_epoch
        self.transfer_epoch = transfer_epoch
        self.device = device

        self.client_num = len(clients)

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
        for i in range(self.pretrain_epoch):
            loss_local = self.train_client(public=False)
            loss_public = self.train_client(public=True)
            print(f"epoch {i}: {loss_local}")
            print(f"epoch {i}: {loss_public}")

        for i in range(self.transfer_epoch):
            self.server.update()
            self.server.distribute()

            for j, client in enumerate(self.clients):
                consensus_loss = client.approach_consensus()
                print(f"epoch {i}, client {j}: {consensus_loss}")
