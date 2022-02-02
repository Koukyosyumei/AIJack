import copy

import torch
import torch.nn as nn

from ..base_attack import BaseAttacker
from .distance import cossim, l2
from .regularization import label_matching, total_variation


class GradientInversion_Attack(BaseAttacker):
    def __init__(
        self,
        target_model,
        x_shape,
        y_shape=None,
        optimize_label=True,
        num_iteration=100,
        lossfunc=nn.CrossEntropyLoss(),
        distancefunc=l2,
        distancename=None,
        optimizer_class=torch.optim.LBFGS,
        optimizername=None,
        tv_coef=0,
        lm_coef=0,
        group_coef=0,
        device="cpu",
        log_interval=10,
        seed=0,
        **kwargs,
    ):
        super().__init__(target_model)
        self.x_shape = x_shape
        self.y_shape = (
            list(target_model.parameters())[-1].shape[0] if y_shape is None else y_shape
        )
        self.optimize_label = optimize_label

        self.num_iteration = num_iteration
        self.lossfunc = lossfunc
        self.distancefunc = distancefunc
        self._setup_distancefunc(distancename)
        self.optimizer_class = optimizer_class
        self._setup_optimizer_class(optimizername)

        self.tv_coef = tv_coef
        self.lm_coef = lm_coef
        self.group_coef = group_coef

        self.device = device
        self.log_interval = log_interval
        self.seed = seed
        self.kwargs = kwargs

        torch.manual_seed(seed)

    def _setup_distancefunc(self, distancename):
        if distancename == "l2":
            self.distancefunc = l2
        elif distancename == "cossim":
            self.distancefunc = cossim

    def _setup_optimizer_class(self, optimizername):
        if optimizername == "LBFGS":
            self.optimizer_class = torch.optim.LBFGS
        elif optimizername == "SGD":
            self.optimizer_class = torch.optim.SGD
        elif optimizername == "Adam":
            self.optimizer_class = torch.optim.Adam

    def _initialize_x(self, batch_size):
        fake_x = torch.randn((batch_size,) + (self.x_shape), requires_grad=True)
        fake_x = fake_x.to(self.device)
        return fake_x

    def _initialize_label(self, batch_size):
        fake_label = torch.randn((batch_size, self.y_shape), requires_grad=True)
        fake_label = fake_label.to(self.device)
        return fake_label

    def _estimate_label(self, received_gradients, batch_size):
        if batch_size == 1:
            fake_label = torch.argmin(torch.sum(received_gradients[-2], dim=1))
        else:
            fake_label = torch.argsort(torch.min(received_gradients[-2], dim=-1)[0])[
                :batch_size
            ]
        fake_label = fake_label.reshape(batch_size)
        fake_label = fake_label.to(self.device)
        return fake_label

    def _setup_closure(self, optimizer, fake_x, fake_label, received_gradients):
        def closure():
            optimizer.zero_grad()
            fake_pred = self.target_model(fake_x)
            if self.optimize_label:
                loss = self.lossfunc(fake_pred, fake_label.softmax(dim=-1))
            else:
                loss = self.lossfunc(fake_pred, fake_label)
            fake_gradients = torch.autograd.grad(
                loss, self.target_model.parameters(), create_graph=True
            )
            distance = self.distancefunc(fake_gradients, received_gradients)

            if self.tv_coef != 0:
                distance += self.tv_coef * total_variation(fake_x)
            if self.lm_coef != 0:
                distance += self.lm_coef * label_matching(fake_pred, fake_label)

            distance.backward(retain_graph=True)
            return distance

        return closure

    def reset_seed(self, seed):
        self.seed = seed
        torch.manual_seed(seed)

    def attack(self, received_gradients, batch_size=1):
        fake_x = self._initialize_x(batch_size)
        fake_label = (
            self._initialize_label(batch_size)
            if self.optimize_label
            else self._estimate_label(received_gradients, batch_size)
        )

        optimizer = (
            self.optimizer_class([fake_x, fake_label], **self.kwargs)
            if self.optimize_label
            else self.optimizer_class(
                [
                    fake_x,
                ],
                **self.kwargs,
            )
        )

        best_distance = float("inf")
        for i in range(self.num_iteration):

            closure = self._setup_closure(
                optimizer, fake_x, fake_label, received_gradients
            )
            distance = optimizer.step(closure)

            if best_distance > distance:
                best_fake_x = copy.deepcopy(fake_x)
                best_fake_label = copy.deepcopy(fake_label)
                best_distance = distance
                best_iteration = i

            if self.log_interval != 0 and i % self.log_interval == 0:
                print(
                    f"iter={i}: {distance}, (best_iter={best_iteration}: {best_distance})"
                )

        return best_fake_x, best_fake_label
