def test_round_decimal():
    import torch

    from aijack.utils import torch_round_x_decimal

    x = torch.Tensor([1.234, 4.567, 6.789])
    x = x.detach()
    x.requires_grad = True

    y = torch_round_x_decimal(x, 2)
    y = torch.sum(torch.Tensor([3.141523]) * y)
    y.backward()

    assert torch.all(x.grad == 3.1400).item()

    x = torch.Tensor([1.234, 4.567, 6.789])
    x = x.detach()
    x.requires_grad = True

    y = torch_round_x_decimal(x, 3)
    y = torch.sum(torch.Tensor([3.141523]) * y)
    y.backward()

    assert torch.all(x.grad == 3.1420).item()
