def test_soteria():
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from aijack.collaborative import FedAvgClient, FedAvgServer
    from aijack.defense import SoteriaManager

    torch.manual_seed(0)

    lr = 0.01
    epochs = 2
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

    manager = SoteriaManager("conv", "lin", target_layer_name="lin.0.weight")
    SoteriaFedAvgClient = manager.attach(FedAvgClient)

    clients = [
        SoteriaFedAvgClient(
            Net(),
            user_id=i,
            lr=lr,
        )
        for i in range(client_num)
    ]
    optimizers = [optim.SGD(client.parameters(), lr=lr) for client in clients]

    global_model = Net()
    server = FedAvgServer(clients, global_model, lr=lr)

    criterion = nn.CrossEntropyLoss()

    loss_log = []
    for _ in range(epochs):
        temp_loss = 0
        for client_idx in range(client_num):
            client = clients[client_idx]
            optimizer = optimizers[client_idx]

            optimizer.zero_grad()
            client.zero_grad()

            outputs = client(x)
            loss = criterion(outputs, y.to(torch.int64))
            client.backward(loss)
            temp_loss = loss.item() / client_num

            optimizer.step()

        loss_log.append(temp_loss)

        server.action(gradients=True)

    assert loss_log[0] > loss_log[1]
