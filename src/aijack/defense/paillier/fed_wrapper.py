import numpy as np

from ...manager import BaseManager
from .torch_wrapper import PaillierTensor


def attach_paillier_to_client_for_encrypted_grad(cls, pk, sk):
    class PaillierClientWrapper(cls):
        def __init__(self, *args, **kwargs):
            super(PaillierClientWrapper, self).__init__(*args, **kwargs)

        def upload_gradients(self):
            pt_grads = super().upload_gradients()
            print("do enc")
            return [
                PaillierTensor(
                    np.vectorize(lambda x: pk.encrypt(x.detach().numpy()))(grad)
                )
                for grad in pt_grads
            ]

        def download(self, model_parameters):
            decrypted_params = {}
            for key, param in model_parameters.items():
                if type(param) == PaillierTensor:
                    decrypted_params[key] = param.decrypt2float(sk)
                else:
                    decrypted_params[key] = param
            return super().download(decrypted_params)

    return PaillierClientWrapper


class PaillierGradientClientManager(BaseManager):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def attach(self, cls):
        return attach_paillier_to_client_for_encrypted_grad(
            cls, *self.args, **self.kwargs
        )
