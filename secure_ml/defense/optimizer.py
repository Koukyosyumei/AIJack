import torch
from torch.optim import Optimizer


class DPSGD(Optimizer):
    """implementation of DPSGD
       reference https://arxiv.org/abs/1607.00133

    Args:
        params: parameters of the model
        lr: learning rate
        sigma: noise scale
        c: grandient norm bound C
    """

    def __init__(self, params, lr=0.05, sigma=0.1, c=0.1):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if sigma < 0.0:
            raise ValueError("Invalid sigma: {}".format(sigma))
        if c < 0.0:
            raise ValueError("Invalid c: {}".format(c))

        defaults = dict(lr=lr, sigma=sigma, c=c)

        super(DPSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DPSGD, self).__setstate__(state)
        pass
        # for group in self.param_groups:
        #    #group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            noise_scale = group["sigma"]
            c = group["c"]

            for p in group['params']:
                if p.grad is None:
                    continue
                # compute gradient
                d_p = p.grad
                # clip gradient
                # torch.clone(d_p).detach()
                clip_grad = torch.clone(d_p).detach(
                ).square().sum().sqrt().div(c).clamp(max=1)
                d_p.div_(clip_grad)
                # add noise
                std = noise_scale * c
                noise = torch.normal(torch.zeros_like(
                    d_p), torch.zeros_like(d_p) + std)
                d_p.add_(noise)
                # descent
                p.add_(d_p, alpha=-1*group['lr'])

        return loss
