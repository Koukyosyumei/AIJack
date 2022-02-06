import torch

from ..base_attack import BaseAttacker


class MI_FACE(BaseAttacker):
    """Implementation of model inversion attack
    reference: https://dl.acm.org/doi/pdf/10.1145/2810103.2813677

    Attributes:
        target_model: model of the victim
        input_shape: input shapes of taregt model
        auxterm_func (function): the default is constant function
        process_func (function): the default is identity function
    """

    def __init__(
        self,
        target_model,
        input_shape,
        auxterm_func=lambda x: 0,
        process_func=lambda x: x,
    ):
        """Inits MI_FACE
        Args:
            target_model: model of the victim
            input_shape: input shapes of taregt model
            auxterm_func (function): the default is constant function
            process_func (function): the default is identity function
        """
        super().__init__(target_model)
        self.input_shape = input_shape
        self.auxterm_func = auxterm_func
        self.process_func = process_func

    def attack(
        self,
        target_label,
        lam,
        num_itr,
    ):
        """Execute the model inversion attack on the target model.

        Args:
            target_label (int): taregt label
            lam (float): step size
            num_itr (int): number of iteration

        Returns:
            x_numpy (np.array) :
            log :
        """
        log = []
        x = torch.zeros(self.input_shape, requires_grad=True)
        for i in range(num_itr):
            c = 1 - self.target_model(x)[:, [target_label]] + self.auxterm_func(x)
            c.backward()
            grad = x.grad
            with torch.no_grad():
                x -= lam * grad
                x = self.process_func(x)
            log.append(c.item())

        x_numpy = x.to("cpu").detach().numpy().copy()
        return x_numpy, log
