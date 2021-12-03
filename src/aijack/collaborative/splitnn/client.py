from ..core import BaseClient


class SplitNNClient(BaseClient):
    def __init__(self, model, user_id=0):
        super().__init__(model, user_id=user_id)
        self.own_intermidiate = None
        self.prev_intermidiate = None
        self.grad_from_next_client = None

    def forward(self, prev_intermediate):
        """Send intermidiate tensor to the server

        Args:
            x (torch.Tensor): the input data

        Returns:
            intermidiate_to_next_client (torch.Tensor): the output of client-side
                                                   model which the client sent
                                                   to the server
        """
        self.prev_intermidiate = prev_intermediate
        self.own_intermidiate = self.model(prev_intermediate)
        intermidiate_to_next_client = self.own_intermidiate.detach().requires_grad_()
        return intermidiate_to_next_client

    def upload(self, x):
        return self.forward(x)

    def download(self, grad_from_next_client):
        self._client_backward(grad_from_next_client)

    def _client_backward(self, grad_from_next_client):
        """Client-side back propagation

        Args:
            grad_from_server: gradient which the server send to the client
        """
        self.grad_from_next_client = grad_from_next_client
        self.own_intermidiate.backward(grad_from_next_client)

    def distribute(self):
        return self.prev_intermidiate.grad.clone()
