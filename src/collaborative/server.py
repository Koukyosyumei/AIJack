import torch


class Server:
    def __init__(self, clients, servre_id=0):
        self.clients = clients
        self.servre_id = servre_id
        self.num_clients = len(clients)
        self.global_model = None

    def update(self):
        pass

    def distribtue(self):
        pass
