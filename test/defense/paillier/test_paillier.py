import numpy as np


def test_paillier_core():
    from aijack.defense.paillier import PaillierKeyGenerator  # noqa: F401

    keygenerator = PaillierKeyGenerator(512)
    pk, sk = keygenerator.generate_keypair()

    ct_1 = pk.encrypt(13)
    assert sk.decrypt2int(ct_1) == 13

    ct_2 = ct_1 * 2
    assert sk.decrypt2int(ct_2) == 26

    ct_3 = ct_1 + 5.6
    np.testing.assert_array_almost_equal(sk.decrypt2float(ct_3), 18.6, decimal=6)

    ct_4 = ct_1 + ct_3
    np.testing.assert_array_almost_equal(sk.decrypt2float(ct_4), 31.6, decimal=6)


def test_paillier_torch():
    import torch  # noqa: F401

    from aijack.defense.paillier import (  # noqa: F401
        PaillierKeyGenerator,
        PaillierTensor,
    )

    keygenerator = PaillierKeyGenerator(512)
    pk, sk = keygenerator.generate_keypair()

    ct_1 = pk.encrypt(13)
    ct_2 = pk.encrypt(0.5)
    ct_3 = ct_1 + ct_2

    pt_1 = PaillierTensor([ct_1, ct_2, ct_3])
    torch.testing.assert_close(
        pt_1.decrypt(sk), torch.Tensor([13, 0.5, 13.5]), atol=1e-5, rtol=1
    )

    pt_2 = pt_1 + torch.Tensor([0.4, 0.1, 0.2])
    torch.testing.assert_close(
        pt_2.decrypt(sk), torch.Tensor([13.4, 0.6, 13.7]), atol=1e-5, rtol=1
    )

    pt_3 = pt_1 * torch.Tensor([1, 2.5, 0.5])
    torch.testing.assert_close(
        pt_3.decrypt(sk), torch.Tensor([13, 1.25, 6.75]), atol=1e-5, rtol=1
    )

    pt_4 = pt_1 - torch.Tensor([0.7, 0.3, 0.6])
    torch.testing.assert_close(
        pt_4.decrypt(sk), torch.Tensor([14.3, 0.2, 12.9]), atol=1e-5, rtol=1
    )

    pt_5 = pt_1 * 2
    torch.testing.assert_close(
        pt_5.decrypt(sk), torch.Tensor([26, 1, 27]), atol=1e-5, rtol=1
    )


def test_pailier_FedAVG():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from aijack.collaborative.fedavg import FedAVGAPI, FedAVGClient, FedAVGServer
    from aijack.defense import PaillierGradientClientManager, PaillierKeyGenerator

    torch.manual_seed(0)

    lr = 0.01
    client_num = 2

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.lin = nn.Sequential(nn.Linear(28 * 28, 10))

        def forward(self, x):
            x = x.reshape((-1, 28 * 28))
            x = self.lin(x)
            return x

    keygenerator = PaillierKeyGenerator(64)
    pk, sk = keygenerator.generate_keypair()

    x = torch.load("test/demodata/demo_mnist_x.pt")
    x.requires_grad = True
    y = torch.load("test/demodata/demo_mnist_y.pt")
    local_dataloaders = [DataLoader(TensorDataset(x, y)) for _ in range(client_num)]

    manager = PaillierGradientClientManager(pk, sk)
    PaillierGradFedAVGClient = manager.attach(FedAVGClient)

    clients = [
        PaillierGradFedAVGClient(Net(), user_id=i, lr=lr, server_side_update=False)
        for i in range(client_num)
    ]
    local_optimizers = [optim.SGD(client.parameters(), lr=lr) for client in clients]

    global_model = Net()
    server = FedAVGServer(clients, global_model, lr=lr, server_side_update=False)

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
