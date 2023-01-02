from ...manager import BaseManager
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


def attach_mpi_to_fedmdserver(cls):
    class MPIFedMDServerWrapper(cls):
        """MPI Wrapper for FedMDServer"""

        def __init__(self, comm, *args, **kwargs):
            super(MPIFedMDServerWrapper, self).__init__(*args, **kwargs)
            self.comm = comm
            self.num_clients = len(self.clients)
            self.round = 0

        def action(self):
            self.mpi_receive()
            self.update()
            self.mpi_distribute()
            self.round += 1

        def mpi_receive(self):
            self.mpi_receive_local_logits()

        def mpi_receive_local_logits(self):
            self.uploaded_logits = []

            while len(self.uploaded_logits) < self.num_clients:
                received_logits = self.comm.recv(tag=LOCAL_LOGIT_TAG)
                self.uploaded_logits.append(received_logits)

        def mpi_distribute(self):
            for client_id in self.clients:
                self.comm.send(self.consensus, dest=client_id, tag=GLOBAL_LOGIT_TAG)

    return MPIFedMDServerWrapper


class MPIFedMDServerManager(BaseManager):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def attach(self, cls):
        return attach_mpi_to_fedmdserver(cls, *self.args, **self.kwargs)
