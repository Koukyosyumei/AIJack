import torch


def _initialize_x(x_shape, batch_size, device):
    """Inits the fake images

    Args:
        batch_size: the batch size

    Returns:
        randomly generated torch.Tensor whose shape is (batch_size, ) + (self.x_shape)
    """
    fake_x = torch.randn((batch_size,) + (x_shape), requires_grad=True, device=device)
    return fake_x


def _initialize_label(y_shape, batch_size, device):
    """Inits the fake labels

    Args:
        batch_size: the batch size

    Returns:
        randomly initialized or estimated labels
    """
    fake_label = torch.randn((batch_size, y_shape), requires_grad=True, device=device)
    fake_label = fake_label.to(device)
    return fake_label


def _estimate_label(received_gradients, batch_size, pos_of_final_fc_layer, device):
    """Estimates the secret labels from the received gradients

    this function is based on the following papers:
    batch_size == 1: https://arxiv.org/abs/2001.02610
    batch_size > 1: https://arxiv.org/abs/2104.07586

    Args:
        received_gradients: gradients received from the client
        batch_size: batch size used to culculate the received_gradients

    Returns:
        estimated labels
    """
    if batch_size == 1:
        fake_label = torch.argmin(
            torch.sum(received_gradients[pos_of_final_fc_layer], dim=1)
        )
    else:
        fake_label = torch.argsort(
            torch.min(received_gradients[pos_of_final_fc_layer], dim=-1)[0]
        )[:batch_size]
    fake_label = fake_label.reshape(batch_size)
    fake_label = fake_label.to(device)
    return fake_label


def _setup_attack(
    x_shape,
    y_shape,
    optimizer_class,
    optimize_label,
    pos_of_final_fc_layer,
    device,
    received_gradients,
    batch_size,
    init_x=None,
    labels=None,
    **kwargs
):
    """Initializes the image and label, and set the optimizer

    Args:
        received_gradients: a list of gradients received from the client
        batch_size: the batch size

    Returns:
        initial images, labels, and the optimizer instance
    """
    fake_x = _initialize_x(x_shape, batch_size, device) if init_x is None else init_x

    if labels is None:
        fake_label = (
            _initialize_label(y_shape, batch_size, device)
            if optimize_label
            else _estimate_label(
                received_gradients,
                batch_size,
                pos_of_final_fc_layer,
                device,
            )
        )
    else:
        fake_label = labels

    optimizer = (
        optimizer_class([fake_x, fake_label], **kwargs)
        if optimize_label
        else optimizer_class(
            [
                fake_x,
            ],
            **kwargs,
        )
    )

    return fake_x, fake_label, optimizer


def _generate_fake_gradients(
    target_model, lossfunc, optimize_label, fake_x, fake_label
):
    fake_pred = target_model(fake_x)
    if optimize_label:
        loss = lossfunc(fake_pred, fake_label.softmax(dim=-1))
    else:
        loss = lossfunc(fake_pred, fake_label)
    fake_gradients = torch.autograd.grad(
        loss,
        target_model.parameters(),
        create_graph=True,
        allow_unused=True,
    )
    return fake_pred, fake_gradients
