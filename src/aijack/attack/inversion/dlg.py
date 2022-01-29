import copy

import torch

from ..base_attack import BaseAttacker


class DLG_Attack(BaseAttacker):
    def __init__(
        self, target_model, x_shape, y_shape, criterion, log_interval=10, seed=0
    ):
        super().__init__(target_model)
        self.criterion = criterion
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.log_interval = log_interval

        torch.manual_seed(seed)

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
                for f_g, c_g in zip(fake_gradients, client_gradients):
                    distance += ((f_g - c_g) ** 2).sum()

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
