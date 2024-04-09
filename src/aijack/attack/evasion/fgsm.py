import torch

from ..base_attack import BaseAttacker


class FGSMAttacker(BaseAttacker):
    """Class implementing the Fast Gradient Sign Method (FGSM) attack.

    This class provides functionality to perform the FGSM attack on a target model.

    Args:
        target_model (torch.nn.Module): The target model to be attacked.
        criterion: The criterion to compute the loss.
        eps (float, optional): The epsilon value for the FGSM attack. Defaults to 0.3.
        grad_lower_bound (float, optional): The lower bound for the gradient. Defaults to -0.1.
        grad_upper_bound (float, optional): The upper bound for the gradient. Defaults to 0.1.
        output_lower_bound (float, optional): The lower bound for the output values. Defaults to -1.0.
        output_upper_bound (float, optional): The upper bound for the output values. Defaults to 1.0.

    Attributes:
        target_model (torch.nn.Module): The target model to be attacked.
        criterion: The criterion to compute the loss.
        eps (float): The epsilon value for the FGSM attack.
        grad_lower_bound (float): The lower bound for the gradient.
        grad_upper_bound (float): The upper bound for the gradient.
        output_lower_bound (float): The lower bound for the output values.
        output_upper_bound (float): The upper bound for the output values.
    """

    def __init__(
        self,
        target_model,
        criterion,
        eps=0.3,
        grad_lower_bound=-0.1,
        grad_upper_bound=0.1,
        output_lower_bound=-1.0,
        output_upper_bound=1.0,
    ):
        super().__init__(target_model)

        self.criterion = criterion
        self.eps = eps
        self.grad_lower_bound = grad_lower_bound
        self.grad_upper_bound = grad_upper_bound
        self.output_lower_bound = output_lower_bound
        self.output_upper_bound = output_upper_bound

    def attack(self, data):
        """Performs the FGSM attack on input seed data.

        Args:
            data (tuple): A tuple containing input seed data and corresponding labels.

        Returns:
            torch.Tensor: The perturbed input data.
        """
        x, y = data
        x.requires_grad = True

        self.target_model.zero_grad()
        output = self.target_model(x)
        loss = self.criterion(output, y)
        loss.backward()
        grad = x.grad.data

        sign_data_grad = grad.sign()
        noise = torch.clamp(
            self.eps * sign_data_grad, self.grad_lower_bound, self.grad_upper_bound
        )
        perturbed_x = x + noise
        perturbed_x = torch.clamp(
            perturbed_x, self.output_lower_bound, self.output_upper_bound
        )
        return perturbed_x
