import torch


def KL_between_normals(mu_q, sigma_q, mu_p, sigma_p):
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q**2, sigma_p**2), dim=1) + torch.sum(
        torch.div(mu_diff_sq, sigma_p**2), dim=1
    )
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5


def mib_loss(
    y,
    sampled_y_pred,
    p_z_given_x_mu,
    p_z_given_x_sigma,
    approximated_z_mean,
    approximated_z_sigma,
    beta=1e-3,
):
    """Implementation of MID loss for NN proposed in https://arxiv.org/abs/2009.05241

    Args:
        y (torch.Tensor): ground-truth label
        sampled_y_pred (torch.Tensor): prdicted output
        p_z_given_x_mu (torch.Tensor): mean of z|x
        p_z_given_x_sigma (torch.Tensor): standard deviation of z|x
        approximated_z_mean (torch.Tensor): approximated mean of z
        approximated_z_sigma (torch.Tensor): approximated standard deviation of z
        beta (float, optional): _description_. weight of I(Z|X) to 1e-3.

    Returns:
        float, torch.Tensor, float: loss values, bound of I(Z|Y) and bound of I(Z|X)
    """

    I_ZX_bound = torch.mean(
        KL_between_normals(
            p_z_given_x_mu, p_z_given_x_sigma, approximated_z_mean, approximated_z_sigma
        )
    )

    loss = torch.nn.CrossEntropyLoss(reduction="none")
    cross_entropy_loss = loss(
        sampled_y_pred, y[:, None].expand(-1, sampled_y_pred.size()[-1])
    )

    cross_entropy_loss_mc = torch.mean(cross_entropy_loss, dim=-1)
    minus_I_ZY_bound = torch.mean(cross_entropy_loss_mc, dim=0)

    return (
        torch.mean(minus_I_ZY_bound + beta * I_ZX_bound),
        minus_I_ZY_bound,
        I_ZX_bound,
    )
