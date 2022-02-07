def test_vib():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from aijack.defense import VIB, mib_loss

    torch.manual_seed(0)

    dim_z = 256
    beta = 1e-3
    batch_size = 1
    samples_amount = 15
    num_epochs = 1

    x = torch.load("test/demodata/demo_mnist_x.pt")
    y = torch.load("test/demodata/demo_mnist_y.pt")
    trainset = TensorDataset(x.view(-1, 28 * 28).float() / 255, y)
    train_loader = DataLoader(trainset, batch_size=batch_size)

    encoder = nn.Sequential(
        nn.Linear(in_features=784, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=2 * dim_z),
    )
    decoder = nn.Linear(in_features=dim_z, out_features=10)

    net = VIB(encoder, decoder, dim_z, num_samples=samples_amount)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    loss_log = []
    minus_izy_log = []
    izx_log = []

    for _ in range(num_epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch
            y_batch = y_batch

            y_pred, result_dict = net(x_batch)
            assert y_pred.shape == (1, 10)
            sampled_y_pred = result_dict["sampled_decoded_outputs"]
            assert sampled_y_pred.shape == (1, 10, 15)
            p_z_given_x_mu = result_dict["p_z_given_x_mu"]
            assert p_z_given_x_mu.shape == (1, 256)
            p_z_given_x_sigma = result_dict["p_z_given_x_sigma"]
            assert p_z_given_x_sigma.shape == (1, 256)

            approximated_z_mean = torch.zeros_like(p_z_given_x_mu)
            approximated_z_sigma = torch.ones_like(p_z_given_x_sigma)

            loss, minus_I_ZY_bound, I_ZX_bound = mib_loss(
                y_batch,
                sampled_y_pred,
                p_z_given_x_mu,
                p_z_given_x_sigma,
                approximated_z_mean,
                approximated_z_sigma,
                beta=beta,
            )

            loss.backward()
            opt.step()
            opt.zero_grad()

            loss_log.append(loss.item())
            minus_izy_log.append(minus_I_ZY_bound.item())
            izx_log.append(I_ZX_bound.item())

    assert loss_log[0] is not None
    assert minus_izy_log[0] is not None
    assert izx_log[0] is not None
