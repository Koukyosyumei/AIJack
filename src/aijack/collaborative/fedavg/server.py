import numpy as np

from ..core import BaseServer


class FedAvgServer(BaseServer):
    def __init__(self, clients, global_model, server_id=0):
        super().__init__(clients, global_model, server_id=server_id)

    def update(self, weight=None):
        if weight is None:
            weight = np.ones(self.num_clients) / self.num_clients

        uploaded_parameters = [c.upload() for c in self.clients]
        averaged_params = uploaded_parameters[0]

        for k in averaged_params.keys():
            for i in range(0, len(uploaded_parameters)):
                local_model_params = uploaded_parameters[i]
                w = weight[i]
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        self.server_model.load_state_dict(averaged_params)

    def distribtue(self):
        for client in self.clients:
            client.download(self.server_model.state_dict())
