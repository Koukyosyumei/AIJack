import copy

import torch

from ...utils import total_variation
from .dlg import DLG_Attack


class GS_Attack(DLG_Attack):
    def __init__(
        self,
        target_model,
        x_shape,
        y_shape,
        criterion,
        log_interval=10,
        seed=0,
        alpha=0,
    ):
        super().__init__(
            target_model,
            x_shape,
            y_shape,
            criterion,
            log_interval=log_interval,
            seed=seed,
        )
        self.alpha = alpha

    def attack(
        self, client_gradients, iteration=100, optimizer=torch.optim.LBFGS, **kwargs
    ):
        fake_x = torch.randn(self.x_shape, requires_grad=True)
        fake_label = torch.randn(self.y_shape).softmax(dim=1).requires_grad_(True)
        optimizer = optimizer([fake_x, fake_label], **kwargs)

        best_fake_x = None
        best_fake_label = None
        best_distance = float("inf")

        for i in range(iteration):

            def closure():
                optimizer.zero_grad()
                loss = self.criterion(
                    self.target_model(fake_x), fake_label.softmax(dim=-1)
                )
                fake_gradients = torch.autograd.grad(
                    loss, self.target_model.parameters(), create_graph=True
                )

                distance = 0
                pnorm_0 = 0
                pnorm_1 = 0
                for f_g, c_g in zip(fake_gradients, client_gradients):
                    pnorm_0 = pnorm_0 + f_g.pow(2).sum()
                    pnorm_1 = pnorm_1 + c_g.pow(2).sum()
                    distance = distance + (f_g * c_g).sum()
                distance = 1 - distance / pnorm_0.sqrt() / pnorm_1.sqrt()
                distance += self.alpha * total_variation(fake_x)

                distance.backward(retain_graph=True)
                return distance

            distance = optimizer.step(closure)
            if best_distance > distance:
                best_fake_x = copy.deepcopy(fake_x)
                best_fake_label = copy.deepcopy(fake_label)
                best_distance = distance
                best_iteration = i

            if i % self.log_interval == 0:
                print(
                    f"iter={i}: {distance}, (best_iter={best_iteration}: {best_distance})"
                )

        return (fake_x, fake_label), (best_fake_x, best_fake_label)
