import random

import torch

from ...manager import BaseManager


def attach_label_flip_attack_to_client(
    cls, victim_label, target_label=None, class_num=None
):
    """Attaches a label flip attack to a client.

    Args:
        cls: The client class.
        victim_label: The label to be replaced.
        target_label: The label to replace the victim label with. If None, a random label will be chosen.
        class_num: The number of classes.

    Returns:
        class: A wrapper class with attached label flip attack.
    """

    class LabelFlipAttackClientWrapper(cls):
        def __init__(self, *args, **kwargs):
            super(LabelFlipAttackClientWrapper, self).__init__(*args, **kwargs)

        def local_train(
            self, local_epoch, criterion, trainloader, optimizer, communication_id=0
        ):
            for i in range(local_epoch):
                running_loss = 0.0
                running_data_num = 0
                for _, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    inputs = inputs.to(self.device)

                    if target_label is not None:
                        labels = torch.where(
                            labels == victim_label, target_label, labels
                        )
                    else:
                        labels = torch.where(
                            labels == victim_label, random.randint(0, class_num), labels
                        )

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

    return LabelFlipAttackClientWrapper


class LabelFlipAttackClientManager(BaseManager):
    def attach(self, cls):
        return attach_label_flip_attack_to_client(cls, *self.args, **self.kwargs)
