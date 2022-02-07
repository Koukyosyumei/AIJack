def test_dpsgd():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import TensorDataset

    from aijack.defense import GeneralMomentAccountant, PrivacyManager

    torch.manual_seed(0)

    lot_size = 1
    batch_size = 1
    iterations = 1
    sigma = 0.5
    l2_norm_clip = 1
    delta = 1e-5

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fla = nn.Flatten()
            self.fc = nn.Linear(784, 10)

        def forward(self, x):
            x = self.fla(x)
            x = self.fc(x)
            x = F.softmax(x, dim=1)
            return x

    x = torch.load("test/demodata/demo_mnist_x.pt")
    y = torch.load("test/demodata/demo_mnist_y.pt")
    trainset = TensorDataset(x.view(-1, 28 * 28).float() / 255, y)

    accountant = GeneralMomentAccountant(
        noise_type="Gaussian",
        search="ternary",
        precision=0.001,
        order_max=1,
        order_min=72,
        max_iterations=1000,
        bound_type="rdp_upperbound_closedformula",
        backend="python",
    )

    privacy_manager = PrivacyManager(
        accountant,
        optim.SGD,
        l2_norm_clip=l2_norm_clip,
        dataset=trainset,
        lot_size=lot_size,
        batch_size=batch_size,
        iterations=iterations,
    )

    accountant.reset_step_info()
    accountant.add_step_info(
        {"sigma": sigma},
        lot_size / len(trainset),
        iterations * (len(trainset) / lot_size),
    )
    estimated_epsilon = accountant.get_epsilon(delta=delta)

    accountant.reset_step_info()
    dpoptimizer_cls, lot_loader, batch_loader = privacy_manager.privatize(
        noise_multiplier=sigma
    )

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = dpoptimizer_cls(net.parameters(), lr=0.05, momentum=0.9)

    for _ in range(iterations):
        running_loss = 0
        data_size = 0
        preds = []
        labels = []
        for data in lot_loader(trainset):
            X_lot, y_lot = data
            optimizer.zero_grad()
            for X_batch, y_batch in batch_loader(TensorDataset(X_lot, y_lot)):
                optimizer.zero_grad_keep_accum_grads()

                pred = net(X_batch)
                loss = criterion(pred, y_batch.to(torch.int64))
                loss.backward()
                optimizer.update_accum_grads()

                running_loss += loss.item()
                data_size += X_batch.shape[0]
                preds.append(pred)
                labels.append(y_batch)

            optimizer.step()

    assert loss.item() is not None
    assert estimated_epsilon is not None
    assert estimated_epsilon == accountant.get_epsilon(delta=delta)
