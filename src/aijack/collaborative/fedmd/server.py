from ..core import BaseServer
from ..core.utils import GLOBAL_LOGIT_TAG, LOCAL_LOGIT_TAG


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

        self.uploaded_logits = []

    def forward(self, x):
        if self.server_model is not None:
            return self.server_model(x)
        else:
            return None

    def action(self):
        self.receive()
        self.update()
        self.distribute()

    def receive(self):
        self.uploaded_logits = [client.upload() for client in self.clients]

    def update(self):
        self.consensus = self.uploaded_logits[0]
        len_clients = len(self.clients)
        for logit in self.uploaded_logits[1:]:
            self.consensus += logit / len_clients

    def distribute(self):
        """Distribute the logits of public dataset to each client."""
        for client in self.clients:
            client.download(self.consensus)


class MPIFedMDServer:
    """MPI Wrapper for FedMDServer

    Args:
        comm: MPI.COMM_WORLD
        server: the instance of FedAvgServer. The `clients` member variable shoud be the list of id.
    """

    def __init__(self, comm, server):
        self.comm = comm
        self.server = server
        self.num_clients = len(self.server.clients)
        self.round = 0

    def __call__(self, *args, **kwargs):
        return self.server(*args, **kwargs)

    def action(self):
        self.mpi_receive()
        self.server.update()
        self.mpi_distribute()
        self.round += 1

    def mpi_receive(self):
        self.mpi_receive_local_logits()

    def mpi_receive_local_logits(self):
        self.server.uploaded_logits = []

        while len(self.server.uploaded_logits) < self.num_clients:
            received_logits = self.comm.recv(tag=LOCAL_LOGIT_TAG)
            self.server.uploaded_logits.append(received_logits)

    def mpi_distribute(self):
        for client_id in self.server.clients:
            self.comm.send(self.server.consensus, dest=client_id, tag=GLOBAL_LOGIT_TAG)
