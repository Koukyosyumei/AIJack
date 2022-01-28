import copy

import torch

from .dlg import DLG_Attack


class GS_Attack(DLG_Attack):
    def attack(
        self, client_gradients, iteration=100, optimizer=torch.optim.LBFGS, **kwargs
    ):
        fake_x = torch.randn(self.x_shape, requires_grad=True)
        fake_label = torch.randn(self.y_shape).softmax(dim=1).requires_grad_(True)
        optimizer = optimizer([fake_x, fake_label], **kwargs)

        best_fake_x = None
        best_fake_label = None
        best_distance = float("inf")

        pnorm = [0, 0]
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
                    pnorm[0] = pnorm[0] + f_g.pow(2).sum()
                    pnorm[1] = pnorm[1] + c_g.pow(2).sum()
                    distance = distance - (f_g * c_g).sum()
                distance = 1 + distance / pnorm[0].sqrt() / pnorm[1].sqrt()

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
