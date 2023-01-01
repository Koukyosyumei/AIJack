import random

import torch

from ...manager import BaseManager


def attach_dba_to_client(
    cls, decomposed_trigger_rules, target_label, poison_ratio, scale
):
    class DistributedBackdoorAttackClientWrapper(cls):
        def __init__(self, *args, **kwargs):
            super(DistributedBackdoorAttackClientWrapper, self).__init__(
                *args, **kwargs
            )

        def upload_gradients(self):
            """Upload the local gradients"""
            gradients = []
            for param, prev_param in zip(self.model.parameters(), self.prev_parameters):
                gradients.append((prev_param - param) / self.lr * scale)
            return gradients

        def local_train(
            self, local_epoch, criterion, trainloader, optimizer, communication_id=0
        ):
            for i in range(local_epoch):
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

                print(
                    f"communication {communication_id}, epoch {i}: client-{self.user_id+1}",
                    running_loss / running_data_num,
                )

    return DistributedBackdoorAttackClientWrapper


class DistributedBackdoorAttackClientManager(BaseManager):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def attach(self, cls):
        return attach_dba_to_client(cls, *self.args, **self.kwargs)
