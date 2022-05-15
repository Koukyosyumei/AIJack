import copy

import torch
import torch.nn as nn

from ...manager import BaseManager
from ..base_attack import BaseAttacker
from .utils.distance import cossim, l2
from .utils.regularization import (
    bn_regularizer,
    group_consistency,
    label_matching,
    total_variance,
)


class GradientInversion_Attack(BaseAttacker):
    """General Gradient Inversion Attacker

    model inversion attack based on gradients can be written as follows:
            x^* = argmin_x' L_grad(x': W, delta_W) + R_aux(x')
    , where X' is the reconstructed image.
    The attacker tries to find images whose gradients w.r.t the given model parameter W
    is similar to the gradients delta_W of the secret images.

    Attributes:
        target_model: a target torch module instance.
        x_shape: the input shape of target_model.
        y_shape: the output shape of target_model.
        optimize_label: If true, only optimize images (the label will be automatically estimated).
        pos_of_final_fc_layer: position of gradients corresponding to the final FC layer
                               within the gradients received from the client.
        num_iteration: number of iterations of optimization.
        optimizer_class: a class of torch optimizer for the attack.
        lossfunc: a function that takes the predictions of the target model and true labels
                  and returns the loss between them.
        distancefunc: a function which takes the gradients of reconstructed images and the client-side gradients
                      and returns the distance between them.
        tv_reg_coef: the coefficient of total-variance regularization.
        lm_reg_coef: the coefficient of label-matching regularization.
        l2_reg_coef: the coefficient of L2 regularization.
        bn_reg_coef: the coefficient of BN regularization.
        gc_reg_coef: the coefficient of group-consistency regularization.
        bn_reg_layers: a list of batch normalization layers of the target model.
        bn_reg_layer_inputs: a lit of extracted inputs of the specified bn layers
        custom_reg_func: a custom regularization function.
        custom_reg_coef: the coefficient of the custom regularization function
        device: device type.
        log_interval: the interval of logging.
        save_loss: If true, save the loss during the attack.
        seed: random state.
        group_num: the size of group,
        group_seed: a list of random states for each worker of the group
        early_stopping: early stopping
    """

    def __init__(
        self,
        target_model,
        x_shape,
        y_shape=None,
        optimize_label=True,
        gradient_ignore_pos=[],
        pos_of_final_fc_layer=-2,
        num_iteration=100,
        optimizer_class=torch.optim.LBFGS,
        optimizername=None,
        lossfunc=nn.CrossEntropyLoss(),
        distancefunc=l2,
        distancename=None,
        tv_reg_coef=0.0,
        lm_reg_coef=0.0,
        l2_reg_coef=0.0,
        bn_reg_coef=0.0,
        gc_reg_coef=0.0,
        bn_reg_layers=[],
        custom_reg_func=None,
        custom_reg_coef=0.0,
        custom_generate_fake_grad_fn=None,
        device="cpu",
        log_interval=10,
        save_loss=True,
        seed=0,
        group_num=5,
        group_seed=None,
        early_stopping=50,
        clamp_range=None,
        **kwargs,
    ):
        """Inits GradientInversion_Attack class.

        Args:
            target_model: a target torch module instance.
            x_shape: the input shape of target_model.
            y_shape: the output shape of target_model.
            optimize_label: If true, only optimize images (the label will be automatically estimated).
            gradient_ignore_pos: a list of positions whihc will be ignored during the culculation of
                                 the distance between gradients
            pos_of_final_fc_layer: position of gradients corresponding to the final FC layer
                                   within the gradients received from the client.
            num_iteration: number of iterations of optimization.
            optimizer_class: a class of torch optimizer for the attack.
            optimizername: a name of optimizer class (priority over optimizer_class).
            lossfunc: a function that takes the predictions of the target model and true labels
                    and returns the loss between them.
            distancefunc: a function which takes the gradients of reconstructed images and the client-side gradients
                        and returns the distance between them.
            distancename: a name of distancefunc (priority over distancefunc).
            tv_reg_coef: the coefficient of total-variance regularization.
            lm_reg_coef: the coefficient of label-matching regularization.
            l2_reg_coef: the coefficient of L2 regularization.
            bn_reg_coef: the coefficient of BN regularization.
            gc_reg_coef: the coefficient of group-consistency regularization.
            bn_reg_layers: a list of batch normalization layers of the target model.
            custom_reg_func: a custom regularization function.
            custom_reg_coef: the coefficient of the custom regularization function
            device: device type.
            log_interval: the interval of logging.
            save_loss: If true, save the loss during the attack.
            seed: random state.
            group_num: the size of group,
            group_seed: a list of random states for each worker of the group
            early_stopping: early stopping
            **kwargs: kwargs for the optimizer
        """
        super().__init__(target_model)
        self.x_shape = x_shape
        self.y_shape = (
            list(target_model.parameters())[-1].shape[0] if y_shape is None else y_shape
        )

        self.optimize_label = optimize_label
        self.gradient_ignore_pos = gradient_ignore_pos
        self.pos_of_final_fc_layer = pos_of_final_fc_layer

        self.num_iteration = num_iteration
        self.lossfunc = lossfunc
        self.distancefunc = distancefunc
        self._setup_distancefunc(distancename)
        self.optimizer_class = optimizer_class
        self._setup_optimizer_class(optimizername)

        self.tv_reg_coef = tv_reg_coef
        self.lm_reg_coef = lm_reg_coef
        self.l2_reg_coef = l2_reg_coef
        self.bn_reg_coef = bn_reg_coef
        self.gc_reg_coef = gc_reg_coef

        self.bn_reg_layers = bn_reg_layers
        self.bn_reg_layer_inputs = {}
        for i, bn_layer in enumerate(self.bn_reg_layers):
            bn_layer.register_forward_hook(self._get_hook_for_input(i))

        self.custom_reg_func = custom_reg_func
        self.custom_reg_coef = custom_reg_coef

        self.custom_generate_fake_grad_fn = custom_generate_fake_grad_fn

        self.device = device
        self.log_interval = log_interval
        self.save_loss = save_loss
        self.seed = seed

        self.group_num = group_num
        self.group_seed = list(range(group_num)) if group_seed is None else group_seed

        self.early_stopping = early_stopping
        self.clamp_range = clamp_range

        self.kwargs = kwargs

        torch.manual_seed(seed)

    def _setup_distancefunc(self, distancename):
        """Assign a function to self.distancefunc according to distancename

        Args:
            distancename: name of the function to culculat the distance between the gradients.
                          currently support 'l2' or 'cossim'.

        Raises:
            ValueError: if distancename is not supported.
        """
        if distancename is None:
            return
        elif distancename == "l2":
            self.distancefunc = l2
        elif distancename == "cossim":
            self.distancefunc = cossim
        else:
            raise ValueError(f"{distancename} is not defined")

    def _setup_optimizer_class(self, optimizername):
        """Assign a class to self.optimizer_class according to optimiername

        Args:
            optimizername: name of optimizer, currently support `LBFGS`, `SGD`, and `Adam`

        Raises:
            ValueError: if optimizername is not supported.
        """
        if optimizername is None:
            return
        elif optimizername == "LBFGS":
            self.optimizer_class = torch.optim.LBFGS
        elif optimizername == "SGD":
            self.optimizer_class = torch.optim.SGD
        elif optimizername == "Adam":
            self.optimizer_class = torch.optim.Adam
        else:
            raise ValueError(f"{optimizername} is not defined")

    def _get_hook_for_input(self, name):
        """Return a hook function to extract the input of the specified layer of the target model

        Args:
            name: the key of self.bn_reg_layer_inputs for the target layer

        Returns:
            hook: a hook function
        """

        def hook(model, inp, output):
            self.bn_reg_layer_inputs[name] = inp[0]

        return hook

    def _initialize_x(self, batch_size):
        """Inits the fake images

        Args:
            batch_size: the batch size

        Returns:
            randomly generated torch.Tensor whose shape is (batch_size, ) + (self.x_shape)
        """
        fake_x = torch.randn(
            (batch_size,) + (self.x_shape), requires_grad=True, device=self.device
        )
        return fake_x

    def _initialize_label(self, batch_size):
        """Inits the fake labels

        Args:
            batch_size: the batch size

        Returns:
            randomly initialized or estimated labels
        """
        fake_label = torch.randn(
            (batch_size, self.y_shape), requires_grad=True, device=self.device
        )
        fake_label = fake_label.to(self.device)
        return fake_label

    def _estimate_label(self, received_gradients, batch_size):
        """Estimate the secret labels from the received gradients

        this function is based on the following papers:
        batch_size == 1: https://arxiv.org/abs/2001.02610
        batch_size > 1: https://arxiv.org/abs/2104.07586

        Args:
            received_gradients: gradients received from the client
            batch_size: batch size used to culculate the received_gradients

        Returns:
            estimated labels
        """
        if batch_size == 1:
            fake_label = torch.argmin(
                torch.sum(received_gradients[self.pos_of_final_fc_layer], dim=1)
            )
        else:
            fake_label = torch.argsort(
                torch.min(received_gradients[self.pos_of_final_fc_layer], dim=-1)[0]
            )[:batch_size]
        fake_label = fake_label.reshape(batch_size)
        fake_label = fake_label.to(self.device)
        return fake_label

    def _culc_regularization_term(
        self, fake_x, fake_pred, fake_label, group_fake_x, received_gradients
    ):
        """Culculate the regularization term

        Args:
            fake_x: reconstructed images
            fake_pred: the predicted value of reconstructed images
            faka_label: the labels of fake_x
            group_fake_x: a list of fake_x of each worker
            received_gradients: gradients received from the client

        Returns:
            culculated regularization term
        """
        reg_term = 0
        if self.tv_reg_coef != 0:
            reg_term += self.tv_reg_coef * total_variance(fake_x)
        if self.lm_reg_coef != 0:
            reg_term += self.lm_reg_coef * label_matching(fake_pred, fake_label)
        if self.l2_reg_coef != 0:
            reg_term += self.l2_reg_coef * torch.norm(fake_x, p=2)
        if self.bn_reg_coef != 0:
            reg_term += self.bn_reg_coef * bn_regularizer(
                self.bn_reg_layer_inputs, self.bn_reg_layers
            )
        if group_fake_x is not None and self.gc_reg_coef != 0:
            reg_term += self.gc_reg_coef * group_consistency(fake_x, group_fake_x)
        if self.custom_reg_func is not None and self.custom_reg_coef != 0:
            context = {
                "attacker": self,
                "fake_x": fake_x,
                "fake_label": fake_label,
                "received_gradients": received_gradients,
                "group_fake_x": group_fake_x,
            }
            reg_term += self.custom_reg_coef * self.custom_reg_func(context)

        return reg_term

    def _generate_fake_gradients(self, fake_x, fake_label):
        fake_pred = self.target_model(fake_x)
        if self.optimize_label:
            loss = self.lossfunc(fake_pred, fake_label.softmax(dim=-1))
        else:
            loss = self.lossfunc(fake_pred, fake_label)
        fake_gradients = torch.autograd.grad(
            loss,
            self.target_model.parameters(),
            create_graph=True,
            allow_unused=True,
        )
        return fake_pred, fake_gradients

    def _setup_closure(
        self, optimizer, fake_x, fake_label, received_gradients, group_fake_x=None
    ):
        """Return a closure function for the optimizer

        Args:
            optimizer (torch.optim.Optimizer): an instance of the optimizer
            fake_x (torch.Tensor): reconstructed images
            fake_label (torch.Tensor): reconstructed or estimated labels
            received_gradients (list): a list of gradients received from the client
            group_fake_x (list, optional): a list of fake_x. Defaults to None.
        """

        def closure():
            if self.custom_generate_fake_grad_fn is None:
                fake_pred, fake_gradients = self._generate_fake_gradients(
                    fake_x, fake_label
                )
            else:
                fake_pred, fake_gradients = self.custom_generate_fake_grad_fn(
                    self, fake_x, fake_label
                )
            optimizer.zero_grad()
            distance = self.distancefunc(
                fake_gradients, received_gradients, self.gradient_ignore_pos
            )
            distance += self._culc_regularization_term(
                fake_x,
                fake_pred,
                fake_label,
                group_fake_x,
                received_gradients,
            )

            distance.backward(retain_graph=True)
            return distance

        return closure

    def _setup_attack(self, received_gradients, batch_size, init_x=None, labels=None):
        """Initialize the image and label, and set the optimizer

        Args:
            received_gradients: a list of gradients received from the client
            batch_size: the batch size

        Returns:
            initial images, labels, and the optimizer instance
        """
        fake_x = self._initialize_x(batch_size) if init_x is None else init_x

        if labels is None:
            fake_label = (
                self._initialize_label(batch_size)
                if self.optimize_label
                else self._estimate_label(received_gradients, batch_size)
            )
        else:
            fake_label = labels

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
        """Reset the random seed

        Args:
            seed (int): the random seed
        """
        self.seed = seed
        torch.manual_seed(seed)

    def attack(
        self,
        received_gradients,
        batch_size=1,
        init_x=None,
        labels=None,
        return_best=True,
    ):
        """Reconstruct the images from the gradients received from the client

        Args:
            received_gradients: the list of gradients received from the client.
            batch_size: batch size.

        Returns:
            a tuple of the best reconstructed images and corresponding labels

        Raises:
            ValueError: If the culculated distance become Nan
        """
        fake_x, fake_label, optimizer = self._setup_attack(
            received_gradients, batch_size, init_x=init_x, labels=labels
        )

        num_of_not_improve_round = 0
        best_distance = float("inf")
        self.log_loss = []
        for i in range(1, self.num_iteration + 1):
            closure = self._setup_closure(
                optimizer, fake_x, fake_label, received_gradients
            )
            distance = optimizer.step(closure)

            if self.clamp_range is not None:
                with torch.no_grad():
                    fake_x[:] = fake_x.clamp(self.clamp_range[0], self.clamp_range[1])

            if torch.sum(torch.isnan(distance)).item():
                raise ValueError("stop because the culculated distance is Nan")

            if self.save_loss:
                self.log_loss.append(distance)

            if best_distance > distance:
                if return_best:
                    best_fake_x = copy.deepcopy(fake_x)
                    best_fake_label = copy.deepcopy(fake_label)
                best_distance = distance
                best_iteration = i
                num_of_not_improve_round = 0
            else:
                num_of_not_improve_round += 1

            if self.log_interval != 0 and i % self.log_interval == 0:
                print(
                    f"iter={i}: {distance}, (best_iter={best_iteration}: {best_distance})"
                )

            if num_of_not_improve_round > self.early_stopping:
                print(
                    f"iter={i}: loss did not improve in the last {self.early_stopping} rounds."
                )
                break

        if return_best:
            return best_fake_x, best_fake_label
        else:
            return fake_x, fake_label

    def group_attack(self, received_gradients, batch_size=1):
        """Multiple simultaneous attacks with different random states

        Args:
            received_gradients: the list of gradients received from the client.
            batch_size: batch size.

        Returns:
            a tuple of the best reconstructed images and corresponding labels

        Raises:
            ValueError: If the culculated distance become Nan
        """
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

        self.log_loss = [[] for _ in range(self.group_num)]
        for i in range(1, self.num_iteration + 1):
            for worker_id in range(self.group_num):
                self.reset_seed(self.group_seed[worker_id])
                closure = self._setup_closure(
                    group_optimizer[worker_id],
                    group_fake_x[worker_id],
                    group_fake_label[worker_id],
                    received_gradients,
                )
                distance = group_optimizer[worker_id].step(closure)

                if self.save_loss:
                    self.log_loss[worker_id].append(distance)

                if best_distance[worker_id] > distance:
                    best_fake_x[worker_id] = copy.deepcopy(group_fake_x[worker_id])
                    best_fake_label[worker_id] = copy.deepcopy(
                        group_fake_label[worker_id]
                    )
                    best_distance[worker_id] = distance
                    best_iteration[worker_id] = i

                if self.log_interval != 0 and i % self.log_interval == 0:
                    print(
                        f"worker_id={worker_id}: iter={i}: {distance}, (best_iter={best_iteration[worker_id]}: {best_distance[worker_id]})"
                    )

        return best_fake_x, best_fake_label


def attach_gradient_inversion_attack_to_server(
    cls,
    x_shape,
    y_shape=None,
    optimize_label=True,
    gradient_ignore_pos=[],
    pos_of_final_fc_layer=-2,
    num_iteration=100,
    optimizer_class=torch.optim.LBFGS,
    optimizername=None,
    lossfunc=nn.CrossEntropyLoss(),
    distancefunc=l2,
    distancename=None,
    tv_reg_coef=0.0,
    lm_reg_coef=0.0,
    l2_reg_coef=0.0,
    bn_reg_coef=0.0,
    gc_reg_coef=0.0,
    bn_reg_layers=[],
    custom_reg_func=None,
    custom_reg_coef=0.0,
    custom_generate_fake_grad_fn=None,
    device="cpu",
    log_interval=10,
    save_loss=True,
    seed=0,
    group_num=5,
    group_seed=None,
    early_stopping=50,
    clamp_range=None,
    target_client_id=0,
    gradinvattack_kwargs={},
):
    class GradientInversionServerWrapper(cls):
        def __init__(self, *args, **kwargs):
            super(GradientInversionServerWrapper, self).__init__(*args, **kwargs)
            self.target_client_id = target_client_id
            self.attacker = GradientInversion_Attack(
                self.server_model,
                x_shape,
                y_shape=y_shape,
                optimize_label=optimize_label,
                gradient_ignore_pos=gradient_ignore_pos,
                pos_of_final_fc_layer=pos_of_final_fc_layer,
                num_iteration=num_iteration,
                optimizer_class=optimizer_class,
                optimizername=optimizername,
                lossfunc=lossfunc,
                distancefunc=distancefunc,
                distancename=distancename,
                tv_reg_coef=tv_reg_coef,
                lm_reg_coef=lm_reg_coef,
                l2_reg_coef=l2_reg_coef,
                bn_reg_coef=bn_reg_coef,
                gc_reg_coef=gc_reg_coef,
                bn_reg_layers=bn_reg_layers,
                custom_reg_func=custom_reg_func,
                custom_reg_coef=custom_reg_coef,
                custom_generate_fake_grad_fn=custom_generate_fake_grad_fn,
                device=device,
                log_interval=log_interval,
                save_loss=save_loss,
                seed=seed,
                group_num=group_num,
                group_seed=group_seed,
                early_stopping=early_stopping,
                clamp_range=clamp_range,
                **gradinvattack_kwargs,
            )

        def change_target_client_id(self, target_client_id):
            self.target_client_id = target_client_id
            self.attacker.target_model = self.clients[self.target_client_id]

        def attack(self, **kwargs):
            received_gradient = self.uploaded_gradients[self.target_client_id]
            return self.attacker.attack(received_gradient, **kwargs)

        def group_attack(self, **kwargs):
            received_gradient = self.uploaded_gradients[self.target_client_id]
            return self.attacker.group_attack(received_gradient, **kwargs)

        def reset_seed(self, seed):
            self.attacker.reset_seed(seed)

    return GradientInversionServerWrapper


class GradientInversionAttackManager(BaseManager):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def attach(self, cls):
        return attach_gradient_inversion_attack_to_server(
            cls, *self.args, **self.kwargs
        )
