def test_FedAVG_sparse_gradient():
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from aijack.collaborative import FedAVGClient, FedAVGServer
    from aijack.defense.sparse import (
        SparseGradientClientManager,
        SparseGradientServerManager,
    )

    torch.manual_seed(0)

    lr = 0.01
    epochs = 2
    client_num = 2

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.lin = nn.Sequential(nn.Linear(28 * 28, 10))

        def forward(self, x):
            x = x.reshape((-1, 28 * 28))
            x = self.lin(x)
            return x

    x = torch.load("test/demodata/demo_mnist_x.pt")
    x.requires_grad = True
    y = torch.load("test/demodata/demo_mnist_y.pt")

    client_manager = SparseGradientClientManager(k=0.3)
    SparseGradientFedAVGClient = client_manager.attach(FedAVGClient)

    server_manager = SparseGradientServerManager()
    SparseGradientFedAVGServer = server_manager.attach(FedAVGServer)

    clients = [
        SparseGradientFedAVGClient(Net(), user_id=i, lr=lr, server_side_update=False)
        for i in range(client_num)
    ]
    optimizers = [optim.SGD(client.parameters(), lr=lr) for client in clients]

    global_model = Net()
    server = SparseGradientFedAVGServer(
        clients, global_model, lr=lr, server_side_update=False
    )

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

        server.action(use_gradients=True)

    assert loss_log[0] > loss_log[1]
