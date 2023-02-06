from torch.utils.data import DataLoader

from .dataloader import LotDataLoader, PoissonSampler
from .dpoptimizer import attach_dpoptimizer
from .adadps import attach_adadps


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
    ):
        self.accountant = accountant
        self.optimizer_cls = optimizer_cls
        self.l2_norm_clip = l2_norm_clip
        self.dataset = dataset
        self.lot_size = lot_size
        self.batch_size = batch_size
        self.iterations = iterations

    def privatize(self, noise_multiplier):
        dpoptimizer_class = attach_dpoptimizer(
            self.optimizer_cls,
            self.accountant,
            self.l2_norm_clip,
            noise_multiplier,
            self.lot_size,
            self.batch_size,
            len(self.dataset),
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
