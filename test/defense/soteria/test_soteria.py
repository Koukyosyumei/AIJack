def test_soteria():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from aijack.collaborative.fedavg import FedAVGAPI, FedAVGClient, FedAVGServer
    from aijack.defense import SoteriaClientManager

    torch.manual_seed(0)

    lr = 0.01
    client_num = 2

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 5),
                nn.Sigmoid(),
                nn.MaxPool2d(3, 3, 1),
                nn.Conv2d(32, 64, 5),
                nn.Sigmoid(),
                nn.MaxPool2d(3, 3, 1),
            )

            self.lin = nn.Sequential(nn.Linear(256, 10))

        def forward(self, x):
            x = self.conv(x)
            x = x.reshape((-1, 256))
            x = self.lin(x)
            return x

    x = torch.load("test/demodata/demo_mnist_x.pt")
    x.requires_grad = True
    y = torch.load("test/demodata/demo_mnist_y.pt")
    local_dataloaders = [DataLoader(TensorDataset(x, y)) for _ in range(client_num)]

    manager = SoteriaClientManager("conv", "lin", target_layer_name="lin.0.weight")
    SoteriaFedAVGClient = manager.attach(FedAVGClient)

    clients = [
        SoteriaFedAVGClient(
            Net(),
            user_id=i,
            lr=lr,
        )
        for i in range(client_num)
    ]
    local_optimizers = [optim.SGD(client.parameters(), lr=lr) for client in clients]

    global_model = Net()
    server = FedAVGServer(clients, global_model, lr=lr)

    criterion = nn.CrossEntropyLoss()

    api = FedAVGAPI(
        server,
        clients,
        criterion,
        local_optimizers,
        local_dataloaders,
        num_communication=2,
        local_epoch=1,
        use_gradients=True,
        custom_action=lambda x: x,
        device="cpu",
    )

    api.run()
