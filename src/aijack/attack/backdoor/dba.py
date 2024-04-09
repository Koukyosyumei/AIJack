import random

import torch

from ...manager import BaseManager


def attach_dba_to_client(
    cls, decomposed_trigger_rules, target_label, poison_ratio, scale
):
    """Wraps the given class in DistributedBackdoorAttackClientWrapper.

    Args:
        cls: Server class
        decomposed_trigger_rules ([function]): list of functions that define the decomposed trigger rules for each client
        target_label (int): a label that the attacker want to make the victim model predict when the inupt contains the trigger
        poison_ratio (float): a ratio of poisoned samples
        scale (_type_): scale for the uploaded gradients

    Returns:
        cls: a class wrapped in DistributedBackdoorAttackClientWrapper
    """

    class DistributedBackdoorAttackClientWrapper(cls):
        """Implementation of https://openreview.net/forum?id=rkgyS0VFvr"""

        def __init__(self, *args, **kwargs):
            super(DistributedBackdoorAttackClientWrapper, self).__init__(
                *args, **kwargs
            )

        def upload_gradients(self):
            """Uploads the local gradients"""
            gradients = []
            for param, prev_param in zip(self.model.parameters(), self.prev_parameters):
                gradients.append((prev_param - param) / self.lr * scale)
            return gradients

        def local_train(
            self, local_epoch, criterion, trainloader, optimizer, communication_id=0
        ):
            loss_log = []

            for _ in range(local_epoch):
                running_loss = 0.0
                running_data_num = 0
                for _, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    inputs = inputs.to(self.device)

                    if random.random() < poison_ratio:
                        inputs = decomposed_trigger_rules[self.user_id](inputs)
                        labels = torch.ones_like(labels) * target_label

                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    self.zero_grad()

                    outputs = self(inputs)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    running_data_num += inputs.shape[0]

                loss_log.append(running_loss / running_data_num)

            return loss_log

    return DistributedBackdoorAttackClientWrapper


class DistributedBackdoorAttackClientManager(BaseManager):
    """Manager class for DistributedBackdoorAttack proposed in https://openreview.net/forum?id=rkgyS0VFvr."""

    def attach(self, cls):
        """Wraps the given class in DistributedBackdoorAttackClientWrapper.

        Returns:
            cls: a class wrapped in DistributedBackdoorAttackClientWrapper
        """
        return attach_dba_to_client(cls, *self.args, **self.kwargs)
