from ..core import BaseClient


class FedAvgClient(BaseClient):
    def __init__(self, model, user_id=0):
        super().__init__(model, user_id=user_id)

    def _upload_parameters(self):
        return self.model.state_dict()

    def _upload_gradients(self):
        gradients = []
        for param, prev_param in zip(self.model.parameters(), self.prev_parameters):
            gradients.append((param.reshape(-1) - prev_param.reshape(-1)).tolist())
        return gradients

    def _download_parameters(self, model_parameters):
        self.model.load_state_dict(model_parameters)
