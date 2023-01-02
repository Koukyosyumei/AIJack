def test_FedAVG_sparse_gradient():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from aijack.collaborative.fedavg import FedAVGAPI, FedAVGClient, FedAVGServer
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
    local_dataloaders = [DataLoader(TensorDataset(x, y)) for _ in range(client_num)]

    client_manager = SparseGradientClientManager(k=0.3)
    SparseGradientFedAVGClient = client_manager.attach(FedAVGClient)

    server_manager = SparseGradientServerManager()
    SparseGradientFedAVGServer = server_manager.attach(FedAVGServer)

    clients = [
        SparseGradientFedAVGClient(Net(), user_id=i, lr=lr, server_side_update=False)
        for i in range(client_num)
    ]
    local_optimizers = [optim.SGD(client.parameters(), lr=lr) for client in clients]

    global_model = Net()
    server = SparseGradientFedAVGServer(
        clients, global_model, lr=lr, server_side_update=False
    )

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
