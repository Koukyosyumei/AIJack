from ..core import BaseServer


class FedMDServer(BaseServer):
    def __init__(
        self,
        clients,
        server_model=None,
        server_id=0,
        device="cpu",
    ):
        super(FedMDServer, self).__init__(clients, server_model, server_id=server_id)
        self.device = device

    def forward(self, x):
        if self.server_model is not None:
            return self.server_model(x)
        else:
            return None

    def action(self):
        self.update()
        self.distribtue()

    def update(self):
        self.consensus = self.clients[0].upload() / len(self.clients)
        for client in self.clients[1:]:
            self.consensus += client.upload() / len(self.clients)

    def distribute(self):
        """Distribute the logits of public dataset to each client."""
        for client in self.clients:
            client.download(self.consensus)
