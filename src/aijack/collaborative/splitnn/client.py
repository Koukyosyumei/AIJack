from ..core import BaseClient


class SplitNNClient(BaseClient):
    def __init__(self, model, user_id=0):
        super().__init__(model, user_id=user_id)
        self.client_side_intermidiate = None
        self.grad_from_server = None

    def forward(self, x):
        """Send intermidiate tensor to the server

        Args:
            x (torch.Tensor): the input data

        Returns:
            intermidiate_to_server (torch.Tensor): the output of client-side
                                                   model which the client sent
                                                   to the server
        """
        self.client_side_intermidiate = self.client_model(x)
        intermidiate_to_server = self.client_side_intermidiate.detach().requires_grad_()
        return intermidiate_to_server

    def upload(self, x):
        return self.forward(x)

    def download(self, grad_from_server):
        self._client_backward(grad_from_server)

    def _client_backward(self, grad_from_server):
        """Client-side back propagation

        Args:
            grad_from_server: gradient which the server send to the client
        """
        self.grad_from_server = grad_from_server
        self.client_side_intermidiate.backward(grad_from_server)
