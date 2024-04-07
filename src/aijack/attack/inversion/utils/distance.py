def l2(fake_gradients, received_gradients, gradient_ignore_pos):
    """Computes the L2 distance between fake and received gradients.

    Args:
        fake_gradients (list of torch.Tensor): List of fake gradients.
        received_gradients (list of torch.Tensor): List of received gradients.
        gradient_ignore_pos (list of int): Positions to ignore while computing distance.

    Returns:
        float: The L2 distance.
    """
    distance = 0
    for i, (f_g, c_g) in enumerate(zip(fake_gradients, received_gradients)):
        if i not in gradient_ignore_pos:
            distance += ((f_g - c_g) ** 2).sum()
    return distance


def cossim(fake_gradients, received_gradients, gradient_ignore_pos):
    """Computes the cosine similarity distance between fake and received gradients.

    Args:
        fake_gradients (list of torch.Tensor): List of fake gradients.
        received_gradients (list of torch.Tensor): List of received gradients.
        gradient_ignore_pos (list of int): Positions to ignore while computing distance.

    Returns:
        float: The cosine similarity distance.
    """
    distance = 0
    pnorm_0 = 0
    pnorm_1 = 0
    for i, (f_g, c_g) in enumerate(zip(fake_gradients, received_gradients)):
        if i not in gradient_ignore_pos:
            pnorm_0 = pnorm_0 + f_g.pow(2).sum()
            pnorm_1 = pnorm_1 + c_g.pow(2).sum()
            distance = distance + (f_g * c_g).sum()
    distance = 1 - distance / pnorm_0.sqrt() / pnorm_1.sqrt()
    return distance
