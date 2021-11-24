from ..core import BaseServer


class SplitNNServer(BaseServer):
    def __init__(self, clients, server_model, server_id):
        super().__init__(clients, server_model, server_id=server_id)
        self.intermidiate_to_server = None
        self.grad_to_client = None

    def forward(self, inputs, client_idx=0):
        # execute client - feed forward network
        self.intermidiate_to_server = self.clients[client_idx](inputs)
        # execute server - feed forward netwoek
        outputs = self.server_forward(self.intermidiate_to_server)

        return outputs

    def backward(self):
        # execute server - back propagation
        grad_to_client = self.server_backward()
        # execute client - back propagation
        self.client.client_backward(grad_to_client)

    def server_forward(self, intermidiate_to_server):
        """server-side prediction

        Args:
            intermidiate_to_server (torch.Tensor): the output of client-side
                                                   model

        Returns:
            outputs (torch.Tensor): outputs of server-side model
        """
        self.intermidiate_to_server = intermidiate_to_server
        outputs = self.server_model(intermidiate_to_server)

        return outputs

    def server_backward(self):
        self.grad_to_client = self.intermidiate_to_server.grad.clone()

    def update(self):
        self.server_backward()

    def distribtue(self):
        return self.grad_to_client
