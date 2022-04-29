def test_fedkd():
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from aijack.collaborative import FedAvgServer, FedKDClient

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
            self.hidden_states = x.reshape((-1, 256))
            x = self.lin(self.hidden_states)
            return x

        def get_hidden_states(self):
            return [self.hidden_states]

    x = torch.load("test/demodata/demo_mnist_x.pt")
    x.requires_grad = True
    y = torch.load("test/demodata/demo_mnist_y.pt")

    criterion = nn.CrossEntropyLoss()
    clients = [
        FedKDClient(
            Net(),
            Net(),
            criterion,
            student_lr=lr,
            teacher_lr=lr,
            adaptive_distillation_losses=True,
            adaptive_hidden_losses=True,
            user_id=i,
        )
        for i in range(client_num)
    ]
    optimizers_students = [optim.SGD(client.parameters(), lr=lr) for client in clients]
    optimizers_teachers = [
        optim.SGD(client.teacher_model.parameters(), lr=lr) for client in clients
    ]

    global_model = Net()
    server = FedAvgServer(clients, global_model, lr=lr)

    teacher_loss_log = []
    student_loss_log = []
    for _ in range(epochs):
        for client_idx in range(client_num):
            client = clients[client_idx]
            optimizer_s = optimizers_students[client_idx]
            optimizer_t = optimizers_teachers[client_idx]

            optimizer_s.zero_grad()
            optimizer_t.zero_grad()

            teacher_loss, student_loss = client.loss(x, y)
            teacher_loss.backward(retain_graph=True)
            student_loss.backward(retain_graph=True)
            optimizer_t.step()
            optimizer_s.step()

            teacher_loss_log.append(teacher_loss.item())
            student_loss_log.append(student_loss.item())

        server.action(use_gradients=True)

    assert teacher_loss_log[1] < teacher_loss_log[0]
    assert student_loss_log[1] < student_loss_log[0]
