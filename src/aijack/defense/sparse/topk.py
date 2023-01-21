import torch

from ...manager import BaseManager


def attach_sparse_gradient_to_client(cls, k):
    """Make the client class communicate the sparse gradients.

    Args:
        cls: client class
        k (int): strength of sparcity

    Returns:
        cls: a class wrapped in SparseGradientClientWrapper
    """

    class SparseGradientClientWrapper(cls):
        def __init__(self, *args, **kwargs):
            super(SparseGradientClientWrapper, self).__init__(*args, **kwargs)

        def upload_gradients(self):
            """Upload sparse gradients"""
            vanila_gradients = super().upload_gradients()
            sparse_gradients = []
            sparse_indices = []
            for vanila_grad in vanila_gradients:
                temp_grad = vanila_grad.reshape(-1)
                # only send top-k gradients
                topk_indices = torch.topk(
                    torch.abs(temp_grad), k=int(len(temp_grad) * k)
                ).indices
                sparse_gradients.append(temp_grad[topk_indices].tolist())
                sparse_indices.append(topk_indices.tolist())

            return sparse_gradients, sparse_indices

    return SparseGradientClientWrapper


def attach_sparse_gradient_to_server(cls):
    """Make the server class communicate the sparse gradients.

    Args:
        cls: server class

    Returns:
        cls: a class wrapped in SparseGradientServerWrapper
    """

    class SparseGradientServerWrapper(cls):
        def __init__(self, *args, **kwargs):
            super(SparseGradientServerWrapper, self).__init__(*args, **kwargs)

        def _preprocess_local_gradients(self, uploaded_grad):
            """Reconstructs dense gradient from the received sparse gradients

            Args:
                uploaded_grad (tuple(torch.Tensor, torch.Tensor)): tuple of non-zero gradients and their positions

            Returns:
                List[torch.Tensor]: list of recovered dense gradients
            """
            sparse_gradients_flattend, sparse_indices = uploaded_grad
            gradients_reshaped = []
            for params, grad, idx in zip(
                self.server_model.parameters(),
                sparse_gradients_flattend,
                sparse_indices,
            ):
                temp_grad = torch.zeros_like(params).reshape(-1)
                temp_grad[idx] = torch.Tensor(grad).to(self.device)
                gradients_reshaped.append(temp_grad.reshape(params.shape))

            return gradients_reshaped

    return SparseGradientServerWrapper


class SparseGradientClientManager(BaseManager):
    """Client-side Manager for sparse gradients."""

    def attach(self, cls):
        return attach_sparse_gradient_to_client(cls, *self.args, **self.kwargs)


class SparseGradientServerManager(BaseManager):
    """Server-side Manager for sparse gradients."""

    def attach(self, cls):
        return attach_sparse_gradient_to_server(cls, *self.args, **self.kwargs)
