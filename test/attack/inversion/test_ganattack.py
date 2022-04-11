def test_ganattack():
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from aijack.attack import attack_ganattack_to_client
    from aijack.collaborative import FedAvgClient, FedAvgServer

    nc = 1
    nz = 100
    ngf = 64
    batch_size = 1
    client_num = 2
    adversary_client_id = 1
    target_label = 3
    fake_batch_size = batch_size
    fake_label = 10

    class Generator(nn.Module):
        def __init__(self, nz, nc, ngf):
            super(Generator, self).__init__()
            # Generator Code (from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 4, 4, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 1),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 1, 1, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        def forward(self, input):
            return self.main(input)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 5),
                nn.Tanh(),
                nn.MaxPool2d(3, 3, 1),
                nn.Conv2d(32, 64, 5),
                nn.Tanh(),
                nn.MaxPool2d(3, 3, 1),
            )

            self.lin = nn.Sequential(
                nn.Linear(256, 200), nn.Tanh(), nn.Linear(200, 11), nn.LogSoftmax()
            )

        def forward(self, x):
            x = self.conv(x)
            x = x.reshape((-1, 256))
            x = self.lin(x)
            return x

    net_1 = Net()
    client_1 = FedAvgClient(net_1, user_id=0)
    optimizer_1 = optim.SGD(
        client_1.parameters(), lr=0.02, weight_decay=1e-7, momentum=0.9
    )

    criterion = nn.CrossEntropyLoss()

    generator = Generator(nz, nc, ngf)
    optimizer_g = optim.SGD(
        generator.parameters(), lr=0.05, weight_decay=1e-7, momentum=0.0
    )
    GANAttackFedAvgClient = attack_ganattack_to_client(
        FedAvgClient,
        target_label,
        generator,
        optimizer_g,
        criterion,
        nz=nz,
    )
    net_2 = Net()
    client_2 = GANAttackFedAvgClient(net_2, user_id=1)
    optimizer_2 = optim.SGD(
        client_2.parameters(), lr=0.02, weight_decay=1e-7, momentum=0.9
    )

    clients = [client_1, client_2]
    optimizers = [optimizer_1, optimizer_2]

    global_model = Net()
    server = FedAvgServer(clients, global_model)

    inputs = torch.load("test/demodata/demo_mnist_x.pt")
    labels = torch.load("test/demodata/demo_mnist_y.pt")

    for epoch in range(2):
        for client_idx in range(client_num):
            client = clients[client_idx]
            optimizer = optimizers[client_idx]

            if epoch != 0 and client_idx == adversary_client_id:
                fake_image = client.attack(fake_batch_size)
                inputs = torch.cat([inputs, fake_image])
                labels = torch.cat(
                    [
                        labels,
                        torch.tensor([fake_label] * fake_batch_size),
                    ]
                )

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = client(inputs)
            loss = criterion(outputs, labels.to(torch.int64))
            client.backward(loss)
            optimizer.step()

        server.action()
