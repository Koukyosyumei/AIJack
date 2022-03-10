def test_m1face():
    import torch.nn as nn

    from aijack.attack import MI_FACE

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
    net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)

    mi = MI_FACE(net, (1, 1, 28, 28), target_label=0, lam=0.1, num_itr=1)
    x_result_1, _ = mi.attack()
    assert x_result_1.shape == (1, 1, 28, 28)
