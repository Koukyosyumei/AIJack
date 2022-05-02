import pytest


def test_sgd():
    import torch  # noqa:F401

    from aijack.collaborative import SGDFLOptimizer  # noqa:F401

    params = [
        torch.nn.Parameter(torch.Tensor([2.5])),
        torch.nn.Parameter(torch.Tensor([1.5])),
    ]
    grads = [torch.Tensor([1.3]), torch.Tensor([0.7])]
    sgd = SGDFLOptimizer(params, lr=0.01, weight_decay=0.0)
    sgd.step(grads)

    assert params[0].data.item() == pytest.approx(2.4870, 1e-6)
    assert params[1].data.item() == pytest.approx(1.4930, 1e-6)
    assert sgd.t == 2

    params = [
        torch.nn.Parameter(torch.Tensor([2.5])),
        torch.nn.Parameter(torch.Tensor([1.5])),
    ]
    grads = [torch.Tensor([1.3]), torch.Tensor([0.7])]
    sgd = SGDFLOptimizer(params, lr=0.01, weight_decay=0.01)
    sgd.step(grads)

    assert params[0].data.item() == pytest.approx(2.48675, 1e-6)
    assert params[1].data.item() == pytest.approx(1.49285, 1e-6)
    assert sgd.t == 2


def test_adam():
    import torch  # noqa:F401

    from aijack.collaborative import AdamFLOptimizer  # noqa:F401

    params = [
        torch.nn.Parameter(torch.Tensor([2.5])),
        torch.nn.Parameter(torch.Tensor([1.5])),
    ]
    grads = [torch.Tensor([1.3]), torch.Tensor([0.7])]
    adam = AdamFLOptimizer(params, lr=0.01, weight_decay=0.00)
    adam.step(grads)

    assert params[0].data.item() == pytest.approx(2.490, 1e-6)
    assert params[1].data.item() == pytest.approx(1.490, 1e-6)
    assert adam.t == 2

    adam.step(grads)
    assert adam.m[0].item() == pytest.approx(0.2470, 1e-6)
    assert params[0].data.item() == pytest.approx(2.480, 1e-6)
