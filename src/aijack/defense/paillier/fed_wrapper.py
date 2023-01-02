import numpy as np

from ...manager import BaseManager
from .torch_wrapper import PaillierTensor


def attach_paillier_to_client_for_encrypted_grad(cls, pk, sk):
    """Makes the client class communicate the encrypted gradients with paillier encryption scheme.

    Args:
        cls: client class
        pk: public key
        sk: secret key
    """

    class PaillierClientWrapper(cls):
        def __init__(self, *args, **kwargs):
            super(PaillierClientWrapper, self).__init__(*args, **kwargs)

        def upload_gradients(self):
            """Uploads encrypted gradients"""
            pt_grads = super().upload_gradients()
            return [
                PaillierTensor(
                    np.vectorize(lambda x: pk.encrypt(x))(grad.detach().numpy())
                )
                for grad in pt_grads
            ]

        def download(self, global_grad):
            """Downloads and decrypt the received global gradients"""
            if not self.initialized:
                # initial parameters are not encrypted
                return super().download(global_grad)
            else:
                decrypted_global_grad = []
                for grad in global_grad:
                    if type(grad) == PaillierTensor:
                        decrypted_global_grad.append(grad.decrypt(sk, self.device))
                    else:
                        decrypted_global_grad.append(grad)
                return super().download(decrypted_global_grad)

    return PaillierClientWrapper


class PaillierGradientClientManager(BaseManager):
    """Client Manager for secure aggregation with Paillier Encryption"""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def attach(self, cls):
        return attach_paillier_to_client_for_encrypted_grad(
            cls, *self.args, **self.kwargs
        )
