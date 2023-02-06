import torch


def _clear_accumulated_grads(opt):
    for group in opt.param_groups:
        for accum_grad in group["accum_grads"]:
            if accum_grad is not None:
                accum_grad.zero_()


def _calculate_clip_coef(opt):
    total_norm = 0.0
    for group in opt.param_groups:
        for param in group["params"]:
            if param.requires_grad:
                total_norm += param.grad.data.norm(2).item() ** 2.0
    total_norm = total_norm**0.5
    clip_coef = min(opt.l2_norm_clip / (total_norm + 1e-6), 1.0)
    return clip_coef


def _apply_clip_coef(opt, clip_coef):
    for group in opt.param_groups:
        for param, accum_grad in zip(group["params"], group["accum_grads"]):
            if param.requires_grad:
                accum_grad.add_(param.grad.data.mul(clip_coef))


def _privatize_lot_grads(opt):
    for group in opt.param_groups:
        for param, accum_grad in zip(group["params"], group["accum_grads"]):
            if param.requires_grad:
                param.grad.data = accum_grad.clone()
                param.grad.data.add_(
                    opt.l2_norm_clip
                    * opt.noise_multiplier
                    * torch.randn_like(param.grad.data)
                )
                param.grad.data.mul_(opt.batch_size / opt.lot_size)


def attach_dpoptimizer(
    cls, accountant, l2_norm_clip, noise_multiplier, lot_size, batch_size, dataset_size
):
    """Wraps the given optimizer class in DPOptimizerWrapper.

    Args:
        accountant (BaseMomentAccountant): moment accountant
        l2_norm_clip (float): upper bound of l2-norm
        noise_multiplier (float): scale for added noise
        lot_size (int): sampled lot size
        batch_size (int): batch size
        dataset_size (int): total number of samples in the dataset

    Raises:
        ValueError: if noise_multiplier < 0.0
        ValueError: if l2_norm_clip < 0

    Returns:
        cls: wrapped DPOptimizerWrapper
    """

    class DPOptimizerWrapper(cls):
        def __init__(self, *args, **kwargs):
            super(DPOptimizerWrapper, self).__init__(*args, **kwargs)

            if noise_multiplier < 0.0:
                raise ValueError(
                    "Invalid noise_multiplier: {}".format(noise_multiplier)
                )
            if l2_norm_clip < 0.0:
                raise ValueError("Invalid l2_norm_clip: {}".format(l2_norm_clip))

            self.accountant = accountant
            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.batch_size = batch_size
            self.lot_size = lot_size

            for group in self.param_groups:
                group["accum_grads"] = [
                    torch.zeros_like(param.data) if param.requires_grad else None
                    for param in group["params"]
                ]

        def zero_grad_keep_accum_grads(self):
            super(DPOptimizerWrapper, self).zero_grad()

        def zero_grad(self):
            self.zero_grad_keep_accum_grads()

        def accumulate_grad(self):
            clip_coef = _calculate_clip_coef(self)
            _apply_clip_coef(self, clip_coef)

        def step(self):
            self.accumulate_grad()

        def zero_grad_for_lot(self):
            _clear_accumulated_grads(self)

        def step_for_lot(self, *args, **kwargs):
            _privatize_lot_grads(self)
            super(DPOptimizerWrapper, self).step(*args, **kwargs)
            accountant.add_step_info(
                {"sigma": self.noise_multiplier}, self.lot_size / dataset_size, 1
            )

    return DPOptimizerWrapper
