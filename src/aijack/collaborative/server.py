import numpy as np
import torch


def transform_list_to_tensor(model_params_list):
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(
            np.asarray(model_params_list[k])
        ).float()
    return model_params_list


class Server:
    def __init__(self, clients, global_model, servre_id=0):
        self.clients = clients
        self.servre_id = servre_id
        self.num_clients = len(clients)
        self.global_model = global_model

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

        self.global_model.load_state_dict(averaged_params)

    def distribtue(self):
        for client in self.clients:
            client.download(self.global_model.state_dict())
