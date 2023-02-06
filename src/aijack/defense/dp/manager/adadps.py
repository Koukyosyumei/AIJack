import torch

from .dpoptimizer import (
    _apply_clip_coef,
    _calculate_clip_coef,
    _clear_accumulated_grads,
    _privatize_lot_grads,
)


def _update_side_info_rmsprop(opt):
    for group in opt.param_groups:
        for param, si in zip(group["params"], group["side_information"]):
            if param.requires_grad:
                si = (opt.beta) * si + (1 - opt.beta) * param.grad.data**2


def _apply_side_infor_rmsprop(opt):
    for group in opt.param_groups:
        for param, si in zip(group["params"], group["side_information"]):
            if param.requires_grad:
                param.grad.data.div_(torch.sqrt(si))


def _update_side_info_adam(opt):
    for group in opt.param_groups:
        for param, si, pm in zip(
            group["params"], group["side_information"], group["potential_momentum"]
        ):
            if param.requires_grad:
                si = (opt.beta) * si + (1 - opt.beta) * param.grad.data**2
                pm = (opt.beta) * pm + (1 - opt.beta) * param.grad.data


def _apply_side_infor_adam(opt):
    for group in opt.param_groups:
        for param, si, pm in zip(
            group["params"], group["side_information"], group["potential_momentum"]
        ):
            if param.requires_grad:
                param.grad.data.mul_(pm / torch.sqrt(si))


def _precondition_grads_with_side_info(opt):
    if opt.mode == "rmsprop":
        _apply_side_infor_rmsprop(opt)
    elif opt.mode == "adam":
        _apply_side_infor_adam(opt)


def attach_adadps(
    cls,
    accountant,
    l2_norm_clip,
    noise_multiplier,
    lot_size,
    batch_size,
    dataset_size,
    mode="rmsprop",
    beta=0.9,
):
    class AdaDPSWrapper(cls):
        """Implementation of AdaDPS proposed in
        `Private Adaptive Optimization with Side information`
        (https://arxiv.org/pdf/2202.05963.pdf)"""

        def __init__(self, *args, **kwargs):
            super(AdaDPSWrapper, self).__init__(*args, **kwargs)

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
            self.mode = mode
            self.beta = beta

            for group in self.param_groups:
                group["accum_grads"] = [
                    torch.zeros_like(param.data) if param.requires_grad else None
                    for param in group["params"]
                ]

            for group in self.param_groups:
                group["side_information"] = [
                    torch.zeros_like(param.data) if param.requires_grad else None
                    for param in group["params"]
                ]

            for group in self.param_groups:
                group["potential_momentum"] = [
                    torch.zeros_like(param.data) if param.requires_grad else None
                    for param in group["params"]
                ]

        def zero_grad(self):
            super(AdaDPSWrapper, self).zero_grad()

        def accumulate_grad(self):
            _precondition_grads_with_side_info(self)
            clip_coef = _calculate_clip_coef(self)
            _apply_clip_coef(self, clip_coef)

        def step_public(self):
            if mode == "rmsprop":
                _update_side_info_rmsprop(self)
            elif mode == "adam":
                _update_side_info_adam(self)

        def step(self):
            self.accumulate_grad()

        def zero_grad_for_lot(self):
            _clear_accumulated_grads(self)

        def step_for_lot(self, *args, **kwargs):
            _privatize_lot_grads(self)
            super(AdaDPSWrapper, self).step(*args, **kwargs)
            accountant.add_step_info(
                {"sigma": self.noise_multiplier}, self.lot_size / dataset_size, 1
            )

    return AdaDPSWrapper
