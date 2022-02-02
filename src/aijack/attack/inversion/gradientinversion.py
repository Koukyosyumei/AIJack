import copy

import torch
import torch.nn as nn

from ..base_attack import BaseAttacker
from .distance import cossim, l2
from .regularization import group_consistency, label_matching, total_variation


class GradientInversion_Attack(BaseAttacker):
    def __init__(
        self,
        target_model,
        x_shape,
        y_shape=None,
        optimize_label=True,
        num_iteration=100,
        optimizer_class=torch.optim.LBFGS,
        optimizername=None,
        lossfunc=nn.CrossEntropyLoss(),
        distancefunc=l2,
        distancename=None,
        tv_coef=0,
        lm_coef=0,
        gc_coef=0,
        custom_reg_func=None,
        custom_reg_coef=0,
        device="cpu",
        log_interval=10,
        seed=0,
        group_num=5,
        group_seed=None,
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
        self.gc_coef = gc_coef

        self.custom_reg_func = custom_reg_func
        self.custom_reg_coef = custom_reg_coef

        self.device = device
        self.log_interval = log_interval
        self.seed = seed

        self.group_num = group_num
        self.group_seed = list(range(group_num)) if group_seed is None else group_seed

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

    def _setup_closure(
        self, optimizer, fake_x, fake_label, received_gradients, group_fake_x=None
    ):
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
            if group_fake_x is not None and self.gc_coef != 0:
                distance += self.gc_coef * group_consistency(fake_x, group_fake_x)
            if self.custom_reg_func is not None and self.custom_reg_coef != 0:
                context = {
                    "attacker": self,
                    "fake_x": fake_x,
                    "fake_label": fake_label,
                    "received_gradients": received_gradients,
                    "group_fake_x": group_fake_x,
                }
                distance += self.custom_reg_coef * self.custom_reg_func(context)

            distance.backward(retain_graph=True)
            return distance

        return closure

    def _setup_attack(self, received_gradients, batch_size):
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

        return fake_x, fake_label, optimizer

    def reset_seed(self, seed):
        self.seed = seed
        torch.manual_seed(seed)

    def attack(self, received_gradients, batch_size=1):
        fake_x, fake_label, optimizer = self._setup_attack(
            received_gradients, batch_size
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

    def gruop_attack(self, received_gradients, batch_size=1):
        group_fake_x = []
        group_fake_label = []
        group_optimizer = []

        for _ in range(self.group_num):
            fake_x, fake_label, optimizer = self._setup_attack(
                received_gradients, batch_size
            )
            group_fake_x.append(fake_x)
            group_fake_label.append(fake_label)
            group_optimizer.append(optimizer)

        best_distance = [float("inf") for _ in range(self.group_num)]
        best_fake_x = copy.deepcopy(group_fake_x)
        best_fake_label = copy.deepcopy(group_fake_label)
        best_iteration = [0 for _ in range(self.group_num)]

        for i in range(self.num_iteration):
            for group_id in range(self.group_num):
                self.reset_seed(self.group_seed[group_id])
                closure = self._setup_closure(
                    group_optimizer[group_id],
                    group_fake_x[group_id],
                    group_fake_label[group_id],
                    received_gradients,
                )
                distance = group_optimizer[group_id].step(closure)

                if best_distance > distance:
                    best_fake_x[group_id] = copy.deepcopy(group_fake_x[group_id])
                    best_fake_label[group_id] = copy.deepcopy(
                        group_fake_label[group_id]
                    )
                    best_distance[group_id] = distance
                    best_iteration[group_id] = i

                if self.log_interval != 0 and i % self.log_interval == 0:
                    print(
                        f"iter={i}: {distance}, (best_iter={best_iteration[group_id]}: {best_distance[group_id]})"
                    )

        return best_fake_x, best_fake_label
