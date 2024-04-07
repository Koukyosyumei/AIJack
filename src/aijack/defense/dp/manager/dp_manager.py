from torch.utils.data import DataLoader

from .adadps import attach_adadps
from .dataloader import LotDataLoader, PoissonSampler
from .dpoptimizer import attach_dpoptimizer


class DPSGDManager:
    """
    Manager class for privatizing DPSGD (Differentially Private Stochastic Gradient Descent) optimization.

    Args:
        accountant: Privacy accountant providing privacy guarantees.
        optimizer_cls: Class of the optimizer to be privatized.
        l2_norm_clip (float): L2 norm clip parameter for gradient clipping.
        dataset: Dataset used for training.
        lot_size (int): Size of the lot (local update).
        batch_size (int): Size of the batch used for training.
        iterations (int): Number of iterations.
        smoothing (bool, optional): Whether to enable smoothing. Defaults to False.
        smoothing_radius (float, optional): Smoothing radius. Defaults to 10.0.

    """

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
        """
        Privatizes the optimizer.

        Args:
            noise_multiplier (float): Noise multiplier for privacy.

        Returns:
            tuple: Tuple containing the privatized optimizer class, lot loader function, and batch loader function.
        """

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
    """
    Manager class for privatizing AdaDPS (Adaptive Differentially Private Stochastic Gradient Descent) optimization.

    Args:
        accountant: Privacy accountant providing privacy guarantees.
        optimizer_cls: Class of the optimizer to be privatized.
        l2_norm_clip (float): L2 norm clip parameter for gradient clipping.
        dataset: Dataset used for training.
        lot_size (int): Size of the lot (local update).
        batch_size (int): Size of the batch used for training.
        iterations (int): Number of iterations.
        mode (str, optional): Mode of optimization (rmsprop or adam). Defaults to "rmsprop".
        beta (float, optional): Beta parameter for optimization. Defaults to 0.9.
        eps_to_avoid_nan (float, optional): Epsilon parameter to avoid NaN during optimization. Defaults to 1e-8.
    """

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
        """
        Privatizes the optimizer.

        Args:
            noise_multiplier (float): Noise multiplier for privacy.

        Returns:
            tuple: Tuple containing the privatized optimizer class, lot loader function, and batch loader function.
        """

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
