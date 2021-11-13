import torch

from ..base_attack import BaseAttacker


class Model_inversion(BaseAttacker):
    def __init__(self, target_model, input_shape):
        """implementation of model inversion attack
           reference https://dl.acm.org/doi/pdf/10.1145/2810103.2813677

        Args:
            target_model: model of the victim
            input_shape: input shapes of taregt model

        Attributes:
            target_model: model of the victim
            input_shape: input shapes of taregt model
        """
        super().__init__(target_model)
        self.input_shape = input_shape

    def attack(self, target_label, lam, num_itr, process_func=lambda x: x):
        """Execute the model inversion attack on the target model.

        Args:
            target_label (int): taregt label
            lam (float) : step size
            num_itr (int) : number of iteration
            process_func (function) : default is identity function

        Returns:
            x_numpy (np.array) :
            loss ([float]) :
        """
        log = []
        x = torch.zeros(self.input_shape, requires_grad=True)
        for i in range(num_itr):
            c = process_func(1 - self.target_model(x)[:, [target_label]])
            c.backward()
            grad = x.grad
            with torch.no_grad():
                x -= lam * grad
            log.append(c.item())

        x_numpy = x.to("cpu").detach().numpy().copy()
        return x_numpy, log
