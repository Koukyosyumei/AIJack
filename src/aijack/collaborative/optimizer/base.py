from abc import abstractmethod


class BaseFLOptimizer:
    """Basic class for optimizers of the server in Federated Learning.

    Args:
        parameters (List[torch.nn.Parameter]): parameters of the model
        lr (float, optional): learning rate. Defaults to 0.01.
        weight_decay (float, optional): coefficient of weight decay. Defaults to 0.0001.
    """

    def __init__(self, parameters, lr=0.01, weight_decay=0.0001):
        self.parameters = list(parameters)
        self.lr = lr
        self.weight_decay = weight_decay
        self.t = 1

    @abstractmethod
    def step(self, grads):
        """Update the parameters with the give gradient

        Args:
            grads (List[torch.Tensor]): list of gradients
        """
        pass
