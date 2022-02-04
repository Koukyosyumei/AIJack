def test_dlg():
    import torch
    import torch.nn as nn

    from aijack.attack import GradientInversion_Attack

    class LeNet(nn.Module):
        def __init__(self, channel=3, hideen=768, num_classes=10):
            super(LeNet, self).__init__()
            act = nn.Sigmoid
            self.body = nn.Sequential(
                nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
                act(),
                nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
                act(),
                nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
                act(),
            )
            self.fc = nn.Sequential(nn.Linear(hideen, num_classes))

        def forward(self, x):
            out = self.body(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

    num_classes = 10
    channel = 1
    hidden = 588
    seed = 42

    x = torch.load("test/demodata/demo_mnist_x.pt")
    y = torch.load("test/demodata/demo_mnist_y.pt")

    criterion = nn.CrossEntropyLoss()
    net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)
    pred = net(x)
    loss = criterion(pred, y)
    received_gradients = torch.autograd.grad(loss, net.parameters())
    received_gradients = [cg.detach() for cg in received_gradients]

    dlg_attacker_1 = GradientInversion_Attack(
        net, (1, 28, 28), lr=1.0, log_interval=0, num_iteration=2, distancename="l2"
    )
    dlg_attacker_1.reset_seed(seed)
    reconstructed_x_1, reconstructed_y_1 = dlg_attacker_1.attack(received_gradients)

    assert reconstructed_x_1.shape == x.shape
    assert dlg_attacker_1.log_loss[1] < dlg_attacker_1.log_loss[0]

    dlg_attacker_2 = GradientInversion_Attack(
        net, (1, 28, 28), lr=1.0, log_interval=0, num_iteration=2, distancename="l2"
    )
    dlg_attacker_2.reset_seed(seed)
    reconstructed_x_2, reconstructed_y_2 = dlg_attacker_2.attack(received_gradients)

    assert torch.sum(reconstructed_x_1 == reconstructed_x_2) == 28 * 28
    assert torch.sum(reconstructed_y_1 == reconstructed_y_2) == 10
    assert dlg_attacker_1.log_loss[1] == dlg_attacker_2.log_loss[1]


def test_gs():
    import torch
    import torch.nn as nn

    from aijack.attack import GradientInversion_Attack

    class LeNet(nn.Module):
        def __init__(self, channel=3, hideen=768, num_classes=10):
            super(LeNet, self).__init__()
            act = nn.Sigmoid
            self.body = nn.Sequential(
                nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
                act(),
                nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
                act(),
                nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
                act(),
            )
            self.fc = nn.Sequential(nn.Linear(hideen, num_classes))

        def forward(self, x):
            out = self.body(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

    num_classes = 10
    channel = 1
    hidden = 588
    seed = 42

    x = torch.load("test/demodata/demo_mnist_x.pt")
    y = torch.load("test/demodata/demo_mnist_y.pt")

    criterion = nn.CrossEntropyLoss()
    net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)
    pred = net(x)
    loss = criterion(pred, y)
    received_gradients = torch.autograd.grad(loss, net.parameters())
    received_gradients = [cg.detach() for cg in received_gradients]

    gs_attacker_1 = GradientInversion_Attack(
        net,
        (1, 28, 28),
        lr=1.0,
        log_interval=0,
        num_iteration=2,
        tv_reg_coef=0.001,
        distancename="cossim",
    )
    gs_attacker_1.reset_seed(seed)
    reconstructed_x_1, reconstructed_y_1 = gs_attacker_1.attack(received_gradients)

    assert reconstructed_x_1.shape == x.shape
    assert gs_attacker_1.log_loss[1] < gs_attacker_1.log_loss[0]

    gs_attacker_2 = GradientInversion_Attack(
        net,
        (1, 28, 28),
        lr=1.0,
        log_interval=0,
        num_iteration=2,
        tv_reg_coef=0.001,
        distancename="cossim",
    )
    gs_attacker_2.reset_seed(seed)
    reconstructed_x_2, reconstructed_y_2 = gs_attacker_2.attack(received_gradients)

    assert torch.sum(reconstructed_x_1 == reconstructed_x_2) == 28 * 28
    assert torch.sum(reconstructed_y_1 == reconstructed_y_2) == 10
    assert gs_attacker_1.log_loss[1] == gs_attacker_2.log_loss[1]


def test_idlg():
    import torch
    import torch.nn as nn

    from aijack.attack import GradientInversion_Attack

    class LeNet(nn.Module):
        def __init__(self, channel=3, hideen=768, num_classes=10):
            super(LeNet, self).__init__()
            act = nn.Sigmoid
            self.body = nn.Sequential(
                nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
                act(),
                nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
                act(),
                nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
                act(),
            )
            self.fc = nn.Sequential(nn.Linear(hideen, num_classes))

        def forward(self, x):
            out = self.body(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

    num_classes = 10
    channel = 1
    hidden = 588
    seed = 42

    x = torch.load("test/demodata/demo_mnist_x.pt")
    y = torch.load("test/demodata/demo_mnist_y.pt")

    criterion = nn.CrossEntropyLoss()
    net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)
    pred = net(x)
    loss = criterion(pred, y)
    received_gradients = torch.autograd.grad(loss, net.parameters())
    received_gradients = [cg.detach() for cg in received_gradients]

    idlg_attacker_1 = GradientInversion_Attack(
        net,
        (1, 28, 28),
        lr=1.0,
        log_interval=0,
        distancename="l2",
        optimize_label=False,
        num_iteration=2,
    )
    idlg_attacker_1.reset_seed(seed)
    reconstructed_x_1, reconstructed_y_1 = idlg_attacker_1.attack(received_gradients)

    assert reconstructed_x_1.shape == x.shape
    assert reconstructed_y_1.item() == 4
    assert idlg_attacker_1.log_loss[1] < idlg_attacker_1.log_loss[0]

    idlg_attacker_2 = GradientInversion_Attack(
        net,
        (1, 28, 28),
        lr=1.0,
        log_interval=0,
        distancename="l2",
        optimize_label=False,
        num_iteration=2,
    )
    idlg_attacker_2.reset_seed(seed)
    reconstructed_x_2, reconstructed_y_2 = idlg_attacker_2.attack(received_gradients)

    assert torch.sum(reconstructed_x_1 == reconstructed_x_2) == 28 * 28
    assert reconstructed_y_1.item() == reconstructed_y_2.item()
    assert idlg_attacker_1.log_loss[1] == idlg_attacker_2.log_loss[1]


def test_cpl():
    import torch
    import torch.nn as nn

    from aijack.attack import GradientInversion_Attack

    class LeNet(nn.Module):
        def __init__(self, channel=3, hideen=768, num_classes=10):
            super(LeNet, self).__init__()
            act = nn.Sigmoid
            self.body = nn.Sequential(
                nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
                act(),
                nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
                act(),
                nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
                act(),
            )
            self.fc = nn.Sequential(nn.Linear(hideen, num_classes))

        def forward(self, x):
            out = self.body(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

    num_classes = 10
    channel = 1
    hidden = 588
    seed = 42

    x = torch.load("test/demodata/demo_mnist_x.pt")
    y = torch.load("test/demodata/demo_mnist_y.pt")

    criterion = nn.CrossEntropyLoss()
    net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)
    pred = net(x)
    loss = criterion(pred, y)
    received_gradients = torch.autograd.grad(loss, net.parameters())
    received_gradients = [cg.detach() for cg in received_gradients]

    cpl_attacker_1 = GradientInversion_Attack(
        net,
        (1, 28, 28),
        lr=1.0,
        log_interval=0,
        distancename="l2",
        optimize_label=False,
        num_iteration=2,
        lm_reg_coef=0.01,
    )
    cpl_attacker_1.reset_seed(seed)
    reconstructed_x_1, reconstructed_y_1 = cpl_attacker_1.attack(received_gradients)

    assert reconstructed_x_1.shape == x.shape
    assert reconstructed_y_1.item() == 4
    assert cpl_attacker_1.log_loss[1] < cpl_attacker_1.log_loss[0]

    cpl_attacker_2 = GradientInversion_Attack(
        net,
        (1, 28, 28),
        lr=1.0,
        log_interval=0,
        distancename="l2",
        optimize_label=False,
        num_iteration=2,
        lm_reg_coef=0.01,
    )
    cpl_attacker_2.reset_seed(seed)
    reconstructed_x_2, reconstructed_y_2 = cpl_attacker_2.attack(received_gradients)

    assert torch.sum(reconstructed_x_1 == reconstructed_x_2) == 28 * 28
    assert reconstructed_y_1.item() == reconstructed_y_2.item()
    assert cpl_attacker_1.log_loss[1] == cpl_attacker_2.log_loss[1]


def test_gradinversion():
    import torch
    import torch.nn as nn

    from aijack.attack import GradientInversion_Attack

    class LeNet(nn.Module):
        def __init__(self, channel=3, hideen=768, num_classes=10):
            super(LeNet, self).__init__()
            act = nn.Sigmoid
            self.body = nn.Sequential(
                nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
                nn.BatchNorm2d(12),
                act(),
                nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
                nn.BatchNorm2d(12),
                act(),
                nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
                nn.BatchNorm2d(12),
                act(),
            )
            self.fc = nn.Sequential(nn.Linear(hideen, num_classes))

        def forward(self, x):
            out = self.body(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

    num_classes = 10
    channel = 1
    hidden = 588
    seed = 42
    group_num = 2

    x = torch.load("test/demodata/demo_mnist_x.pt")
    y = torch.load("test/demodata/demo_mnist_y.pt")

    criterion = nn.CrossEntropyLoss()
    net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)
    pred = net(x)
    loss = criterion(pred, y)
    received_gradients = torch.autograd.grad(loss, net.parameters())
    received_gradients = [cg.detach() for cg in received_gradients]

    gradinversion_attacker_1 = GradientInversion_Attack(
        net,
        (1, 28, 28),
        num_iteration=2,
        lr=1.0,
        log_interval=0,
        distancename="l2",
        optimize_label=False,
        bn_reg_layers=[net.body[1], net.body[4], net.body[7]],
        group_num=group_num,
        tv_reg_coef=0.001,
        l2_reg_coef=0.0001,
        bn_reg_coef=0.001,
        gc_reg_coef=0.001,
    )
    gradinversion_attacker_1.reset_seed(seed)
    reconstructed_x_1, reconstructed_y_1 = gradinversion_attacker_1.group_attack(
        received_gradients, batch_size=1
    )

    for gid in range(group_num):
        assert reconstructed_x_1[gid].shape == x.shape
        assert reconstructed_y_1[gid].item() == 4
        assert (
            gradinversion_attacker_1.log_loss[gid][1]
            < gradinversion_attacker_1.log_loss[gid][0]
        )
        assert (
            torch.sum(reconstructed_x_1[gid - 1] == reconstructed_x_1[gid]) != 28 * 28
        )
