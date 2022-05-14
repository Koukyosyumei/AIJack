import torch

from .base import BaseFLOptimizer


class AdamFLOptimizer(BaseFLOptimizer):
    """Implementation of Adam to update the global model of Federated Learning.

    Args:
        parameters (List[torch.nn.Parameter]): parameters of the model
        lr (float, optional): learning rate. Defaults to 0.01.
        weight_decay (float, optional): coefficient of weight decay. Defaults to 0.0001.
        beta1 (float, optional): 1st-order exponential decay. Defaults to 0.9.
        beta2 (float, optional): 2nd-order exponential decay. Defaults to 0.999.
        epsilon (float, optional): a small value to prevent zero-devision. Defaults to 1e-8.
    """

    def __init__(
        self,
        parameters,
        lr=0.01,
        weight_decay=0.0001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        super().__init__(parameters, lr=lr, weight_decay=weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = [torch.zeros_like(param.data) for param in self.parameters]
        self.v = [torch.zeros_like(param.data) for param in self.parameters]

    def step(self, grads):
        """Update the parameters with the give gradient

        Args:
            grads (List[torch.Tensor]): list of gradients
        """
        for i, (param, grad) in enumerate(zip(self.parameters, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            param.data -= self.lr * (
                m_hat / torch.sqrt(v_hat) + self.weight_decay * param.data
            )
        self.t += 1
