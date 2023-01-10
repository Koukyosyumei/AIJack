import numpy as np
import torch


def transform_list_to_tensor(model_params_list):
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(
            np.asarray(model_params_list[k])
        ).float()
    return model_params_list


def default_local_train_for_client(
    self, local_epoch, criterion, trainloader, optimizer
):
    running_loss = 0.0
    for _ in range(local_epoch):
        for data in trainloader:
            _, x, y = data
            x = x.to(self.device)
            y = y.to(self.device).to(torch.int64)

            optimizer.zero_grad()
            loss = criterion(self(x), y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    return running_loss
