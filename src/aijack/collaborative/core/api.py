from abc import abstractmethod


class BaseFLKnowledgeDistillationAPI:
    def __init__(
        self,
        server,
        clients,
        public_dataloader,
        local_dataloaders,
        validation_dataloader,
        criterion,
        num_commnication=10,
        device="cpu",
    ):
        self.server = server
        self.clients = clients
        self.public_dataloader = public_dataloader
        self.local_dataloaders = local_dataloaders
        self.validation_dataloader = validation_dataloader
        self.criterion = criterion
        self.num_commnication = num_commnication
        self.device = device

        self.client_num = len(clients)

    @abstractmethod
    def run(self):
        pass
