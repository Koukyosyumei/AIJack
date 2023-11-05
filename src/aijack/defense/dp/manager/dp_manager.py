from torch.utils.data import DataLoader

from .adadps import attach_adadps
from .dataloader import LotDataLoader, PoissonSampler
from .dpoptimizer import attach_dpoptimizer


class DPSGDManager:
    def __init__(
        self,
        accountant,
        optimizer_cls,
        l2_norm_clip,
        dataset,
        lot_size,
        batch_size,
        iterations,
        smoothing=False,
        smoothing_radius=10.0,
    ):
        self.accountant = accountant
        self.optimizer_cls = optimizer_cls
        self.l2_norm_clip = l2_norm_clip
        self.dataset = dataset
        self.lot_size = lot_size
        self.batch_size = batch_size
        self.iterations = iterations
        self.smoothing = smoothing
        self.smoothing_radius = smoothing_radius

    def privatize(self, noise_multiplier):
        dpoptimizer_class = attach_dpoptimizer(
            self.optimizer_cls,
            self.accountant,
            self.l2_norm_clip,
            noise_multiplier,
            self.lot_size,
            self.batch_size,
            len(self.dataset),
            self.smoothing,
            self.smoothing_radius,
        )

        def lot_loader(dp_optimizer):
            return LotDataLoader(
                dp_optimizer,
                self.dataset,
                batch_sampler=PoissonSampler(
                    self.dataset, self.lot_size, self.iterations
                ),
            )

        def batch_loader(lot):
            return DataLoader(lot, batch_size=self.batch_size, drop_last=True)

        return dpoptimizer_class, lot_loader, batch_loader


class AdaDPSManager:
    def __init__(
        self,
        accountant,
        optimizer_cls,
        l2_norm_clip,
        dataset,
        lot_size,
        batch_size,
        iterations,
        mode="rmsprop",
        beta=0.9,
        eps_to_avoid_nan=1e-8,
    ):
        self.accountant = accountant
        self.optimizer_cls = optimizer_cls
        self.l2_norm_clip = l2_norm_clip
        self.dataset = dataset
        self.lot_size = lot_size
        self.batch_size = batch_size
        self.iterations = iterations
        self.mode = mode
        self.beta = beta
        self.eps_to_avoid_nan = eps_to_avoid_nan

    def privatize(self, noise_multiplier):
        dpoptimizer_class = attach_adadps(
            self.optimizer_cls,
            self.accountant,
            self.l2_norm_clip,
            noise_multiplier,
            self.lot_size,
            self.batch_size,
            len(self.dataset),
            self.mode,
            self.beta,
            self.eps_to_avoid_nan,
        )

        def lot_loader(dp_optimizer):
            return LotDataLoader(
                dp_optimizer,
                self.dataset,
                batch_sampler=PoissonSampler(
                    self.dataset, self.lot_size, self.iterations
                ),
            )

        def batch_loader(lot):
            return DataLoader(lot, batch_size=self.batch_size, drop_last=True)

        return dpoptimizer_class, lot_loader, batch_loader
