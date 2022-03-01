from ..core import BaseServer


class FedMDServer(BaseServer):
    def __init__(
        self,
        clients,
        server_id=0,
        device="cpu",
    ):
        super(FedMDServer, self).__init__(clients, None, server_id=server_id)
        self.device = device

    def action(self):
        self.update()
        self.distribtue()

    def update(self):
        self.consensus = self.clients[0].upload() / len(self.clients)
        for client in self.clients[1:]:
            self.consensus += client.upload() / len(self.clients)

    def distribtue(self):
        """Distribute the logits of public dataset to each client."""
        for client in self.clients:
            client.download(self.consensus)
