import torch

from ..base_attack import BaseAttacker


class FGSMAttacker(BaseAttacker):
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

    def attack(self, x):
        x.requires_grad = True

        self.target_model.zero_grad()
        output = self.target_model(x)
        loss = self.criterion(output, x)
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
