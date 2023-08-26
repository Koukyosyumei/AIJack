import torch

from ..base_attack import BaseAttacker


class DIVAWhiteBoxAttacker(BaseAttacker):
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
