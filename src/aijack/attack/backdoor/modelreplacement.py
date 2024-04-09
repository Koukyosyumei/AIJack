import torch

from ...manager import BaseManager


def l2norm_checker(client):
    l2 = torch.tensor(0.0, requires_grad=True)
    for param, prev_param in zip(client.model.parameters(), client.prev_parameters):
        l2 = l2 + torch.norm(param - prev_param, 2)
    return l2


def attach_modelreplacement_to_client(
    cls,
    alpha,
    gamma,
    criterion_anomaly_detection=l2norm_checker,
    reference_dataloader=None,
    eps=1e-6,
):
    """Wraps the given class in ModelReplacementAttackClientWrapper.

    Args:
        cls: Client class
    Returns:
        cls: a class wrapped in ModelReplacementAttackClientWrapper
    """

    class ModelReplacementAttackClientWrapper(cls):
        """Implementation of https://proceedings.mlr.press/v108/bagdasaryan20a/bagdasaryan20a.pdf"""

        def __init__(self, *args, **kwargs):
            super(ModelReplacementAttackClientWrapper, self).__init__(*args, **kwargs)

        def upload_gradients(self):
            """Uploads the local gradients"""
            gradients = []
            for param, prev_param in zip(self.model.parameters(), self.prev_parameters):
                gradients.append(gamma * (prev_param - param) / (self.lr))
            return gradients

        def local_train(
            self, local_epoch, criterion, trainloader, optimizer, communication_id=0
        ):
            loss_log = []

            for _ in range(local_epoch):
                if reference_dataloader is not None:
                    running_loss = 0.0
                    running_data_num = 0
                    with torch.no_grad():
                        for data in reference_dataloader:
                            inputs, labels = data
                            inputs = inputs.to(self.device)
                            labels = labels.to(self.labels)
                            outputs = self(inputs)

                            loss = criterion(outputs, labels)
                            running_loss += loss.item()
                            running_data_num += inputs.shape()[0]

                        if running_loss / running_data_num <= eps:
                            break

                running_loss = 0.0
                running_data_num = 0
                for _, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    self.zero_grad()

                    outputs = self(inputs)
                    loss = alpha * criterion(outputs, labels)
                    loss += (1 - alpha) * criterion_anomaly_detection(self)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    running_data_num += inputs.shape[0]

                loss_log.append(running_loss / running_data_num)

            return loss_log

    return ModelReplacementAttackClientWrapper


class ModelReplacementAttackClientManager(BaseManager):
    """Manager class for DistributedBackdoorAttack proposed in
    https://proceedings.mlr.press/v108/bagdasaryan20a/bagdasaryan20a.pdf."""

    def attach(self, cls):
        """Wraps the given class in ModelReplacementAttackClientWrapper.

        Returns:
            cls: a class wrapped in ModelReplacementAttackClientWrapper
        """
        return attach_modelreplacement_to_client(cls, *self.args, **self.kwargs)
