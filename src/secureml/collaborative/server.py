import numpy as np
import torch


def transform_list_to_tensor(model_params_list):
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(
            np.asarray(model_params_list[k])
        ).float()
    return model_params_list


class Server:
    def __init__(self, clients, servre_id=0):
        self.clients = clients
        self.servre_id = servre_id
        self.num_clients = len(clients)
        self.global_model = None

    def update(self, uploaded_parameters, training_nums):
        total_training_data_size = sum(training_nums)
        averaged_params = uploaded_parameters[0]
        for k in averaged_params.keys():
            for i in range(0, len(uploaded_parameters)):
                local_model_params = uploaded_parameters[i]
                local_sample_number = training_nums[i]
                w = local_sample_number / total_training_data_size
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        self.global_model.load_state_dict(averaged_params)

    def distribtue(self):
        return self.global_model.cpu().state_dict()
