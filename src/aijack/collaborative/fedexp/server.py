import torch

from ..fedavg import FedAVGServer


class FedEXPServer(FedAVGServer):
    """Implementation of 'Jhunjhunwala, Divyansh, Shiqiang Wang, and Gauri Joshi. "FedExP: Speeding up Federated Averaging Via Extrapolation." arXiv preprint arXiv:2301.09604 (2023).'"""

    def __init__(self, *args, eps=1e-5, **kwargs):
        super(FedEXPServer, self).__init__(*args, **kwargs)
        self.eps = eps

    def update(self, *args, **kwargs):
        self.update_from_gradients()

    def update_from_gradients(self):
        self.aggregated_gradients = [
            torch.zeros_like(params) for params in self.server_model.parameters()
        ]
        grad_norms = []

        M = len(self.uploaded_gradients)
        len_gradients = len(self.aggregated_gradients)

        for i, gradients in enumerate(self.uploaded_gradients):
            for gradient_id in range(len_gradients):
                self.aggregated_gradients[gradient_id] = (
                    gradients[gradient_id] * (1 / M)
                    + self.aggregated_gradients[gradient_id]
                )
                grad_norms.append(
                    torch.sqrt(sum([torch.sum(g * g) for g in gradients[gradient_id]]))
                )

        agg_grad_norm = torch.sqrt(
            sum([torch.sum(g * g) for g in self.aggregated_gradients])
        )
        lr = max(1, sum([g / (2 * M * (agg_grad_norm + self.eps)) for g in grad_norms]))

        self.optimizer.step(self.aggregated_gradients)
