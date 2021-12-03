from ..core import BaseClient


class FedAvgClient(BaseClient):
    def __init__(self, model, user_id=0):
        super().__init__(model, user_id=user_id)

    def upload(self):
        return self.model.state_dict()

    def download(self, model_parameters):
        self.model.load_state_dict(model_parameters)
