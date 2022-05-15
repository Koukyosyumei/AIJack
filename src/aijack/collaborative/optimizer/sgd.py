from .base import BaseFLOptimizer


class SGDFLOptimizer(BaseFLOptimizer):
    """Implementation of SGD to update the global model of Federated Learning.

    Args:
        parameters (List[torch.nn.Parameter]): parameters of the model
        lr (float, optional): learning rate. Defaults to 0.01.
        weight_decay (float, optional): coefficient of weight decay. Defaults to 0.0001.
    """

    def __init__(self, parameters, lr=0.01, weight_decay=0.0000):
        super().__init__(parameters, lr=lr, weight_decay=weight_decay)

    def step(self, grads):
        """Update the parameters with the give gradient

        Args:
            grads (List[torch.Tensor]): list of gradients
        """
        for param, grad in zip(self.parameters, grads):
            param.data -= self.lr * (grad + self.weight_decay * param.data)
        self.t += 1
