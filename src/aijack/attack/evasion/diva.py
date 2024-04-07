import torch

from ..base_attack import BaseAttacker


class DIVAWhiteBoxAttacker(BaseAttacker):
    """Class implementing the DIVA white-box attack.

    This class provides functionality to perform the DIVA white-box attack on a target model.

    Args:
        target_model (torch.nn.Module): The target model to be attacked.
        target_model_on_edge (torch.nn.Module): The target model deployed on the edge.
        c (float, optional): The trade-off parameter between origin and edge predictions. Defaults to 1.0.
        num_itr (int, optional): The number of iterations for the attack. Defaults to 1000.
        eps (float, optional): The maximum perturbation allowed. Defaults to 0.1.
        lam (float, optional): The step size for gradient updates. Defaults to 0.01.
        device (str, optional): The device to perform computation on. Defaults to "cpu".

    Attributes:
        target_model (torch.nn.Module): The target model to be attacked.
        target_model_on_edge (torch.nn.Module): The target model deployed on the edge.
        c (float): The trade-off parameter between origin and edge predictions.
        num_itr (int): The number of iterations for the attack.
        eps (float): The maximum perturbation allowed.
        lam (float): The step size for gradient updates.
        device (str): The device to perform computation on.

    """

    def __init__(
        self,
        target_model,
        target_model_on_edge,
        c=1.0,
        num_itr=1000,
        eps=0.1,
        lam=0.01,
        device="cpu",
    ):
        super().__init__(target_model)
        self.target_model_on_edge = target_model_on_edge
        self.c = c
        self.num_itr = num_itr
        self.eps = eps
        self.lam = lam
        self.device = device

    def attack(self, data):
        """Performs the DIVA white-box attack on input data.

        Args:
            data (tuple): A tuple containing input data and corresponding labels.

        Returns:
            tuple: A tuple containing the adversarial examples and attack logs.

        """

        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)
        x_origin = torch.clone(x)

        log_loss = []
        log_perturbation = []

        for _ in range(self.num_itr):
            x = x.detach().to(self.device)
            x.requires_grad = True
            origin_pred = self.target_model(x)
            edge_pred = self.target_model_on_edge(x)
            loss = origin_pred[:, y] - self.c * edge_pred[:, y]
            loss.backward()
            grad = x.grad

            with torch.no_grad():
                x += self.lam * grad
                x = torch.clamp(x, x_origin - self.eps, x_origin + self.eps)

                log_loss.append(loss.item())
                log_perturbation.append(torch.mean((x - x_origin).abs()).item())

                if origin_pred.argmax().item() != edge_pred.argmax().item():
                    break

        return x, {"log_loss": log_loss, "log_perturbation": log_perturbation}
