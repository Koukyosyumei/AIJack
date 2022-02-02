import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from aijack.attack import GradientInversion_Attack
from aijack.collaborative import FedAvgClient, FedAvgServer
from aijack.defense import SetoriaFedAvgClient
from aijack.utils import NumpyDataset

lr = 0.01
epochs = 1
batch_size = 1
test_batch_size = 16
client_num = 2
local_data_num = 1
local_label_num = 1
setoria = True


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(3, 3, 1),
            nn.Conv2d(32, 64, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(3, 3, 1),
        )

        self.lin = nn.Sequential(nn.Linear(256, 10))

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape((-1, 256))
        x = self.lin(x)
        return x


def prepare_dataloaders_noniid(client_num=2, local_label_num=2, local_data_num=20):
    at_t_dataset_train = torchvision.datasets.MNIST(
        root="MNIST/.", train=True, download=True
    )
    at_t_dataset_test = torchvision.datasets.MNIST(
        root="MNIST/.", train=False, download=True
    )

    X = at_t_dataset_train.train_data.numpy()
    y = at_t_dataset_train.train_labels.numpy()

    # ToTensor：画像のグレースケール化（RGBの0~255を0~1の範囲に正規化）、Normalize：Z値化（RGBの平均と標準偏差を0.5で決め打ちして正規化）
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    test_set = NumpyDataset(
        at_t_dataset_test.test_data.numpy(),
        at_t_dataset_test.test_labels.numpy(),
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=True, num_workers=0
    )

    trainloaders = []
    train_sizes = []
    idx_used = []
    for c in range(client_num):
        assigned_labels = random.sample(range(10), local_label_num)
        print(c, assigned_labels)
        idx = np.concatenate([np.where(y == al)[0] for al in assigned_labels])
        assigned_idx = random.sample(list(set(idx) - set(idx_used)), local_data_num)

        temp_trainset = NumpyDataset(
            X[assigned_idx], y[assigned_idx], transform=transform
        )
        temp_trainloader = torch.utils.data.DataLoader(
            temp_trainset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        trainloaders.append(temp_trainloader)
        train_sizes.append(len(temp_trainset))

        idx_used += assigned_idx

    assert len(idx_used) == len(list(set(idx_used)))

    return X, y, trainloaders, testloader, train_sizes, idx_used


def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    (
        X,
        y,
        trainloaders,
        global_trainloader,
        dataset_nums,
        _,
    ) = prepare_dataloaders_noniid(
        client_num=client_num,
        local_label_num=local_label_num,
        local_data_num=local_data_num,
    )

    criterion = nn.CrossEntropyLoss()

    if not setoria:
        clients = [
            FedAvgClient(Net(), user_id=i, lr=lr).to(device) for i in range(client_num)
        ]
    else:
        clients = [
            SetoriaFedAvgClient(Net(), "conv", "lin", user_id=i, lr=lr).to(device)
            for i in range(client_num)
        ]

    optimizers = [optim.SGD(client.parameters(), lr=lr) for client in clients]

    global_model = Net()
    global_model.to(device)
    server = FedAvgServer(clients, global_model, lr=lr)

    for epoch in range(epochs):
        for client_idx in range(client_num):
            client = clients[client_idx]
            trainloader = trainloaders[client_idx]
            optimizer = optimizers[client_idx]

            running_loss = 0.0
            for _, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                inputs.requires_grad = True
                labels = labels.to(device)

                optimizer.zero_grad()
                client.zero_grad()

                outputs = client(inputs)
                loss = criterion(outputs, labels.to(torch.int64))

                if not setoria:
                    loss.backward()
                else:
                    client.action_before_lossbackward()
                    loss.backward()
                    client.action_after_lossbackward("lin.0.weight")

                optimizer.step()

                running_loss += loss.item()
            print(
                f"epoch {epoch}: client-{client_idx+1}",
                running_loss / dataset_nums[client_idx],
            )

        server.action(gradients=True)

    client_gradients = server.uploaded_gradients[-1]
    client_gradients = [(c / len(trainloader)).detach() for c in client_gradients]

    attacker = GradientInversion_Attack(
        server,
        (1, 28, 28),
        lr=1.0,
        log_interval=0,
        num_iteration=100,
        distancename="l2",
    )
    num_seeds = 5
    fig = plt.figure()
    for s in range(num_seeds):
        attacker.reset_seed(s)
        result = attacker.attack(client_gradients)
        ax1 = fig.add_subplot(2, num_seeds, s + 1)
        ax1.axis("off")
        ax1.imshow(result[0].detach().numpy()[0][0], cmap="gray")
        ax1.set_title(torch.argmax(result[1]).item())
        ax2 = fig.add_subplot(2, num_seeds, num_seeds + s + 1)
        ax2.imshow(cv2.medianBlur(result[0].detach().numpy()[0][0], 5), cmap="gray")
        ax2.axis("off")
    plt.suptitle("Result of DLG")
    plt.tight_layout()
    plt.savefig("dlg.png")
    plt.close()

    attacker = GradientInversion_Attack(
        server,
        (1, 28, 28),
        lr=1.0,
        log_interval=0,
        num_iteration=100,
        distancename="cossim",
    )
    num_seeds = 5
    fig = plt.figure()
    for s in range(num_seeds):
        attacker.reset_seed(s)
        result = attacker.attack(client_gradients)
        ax1 = fig.add_subplot(2, num_seeds, s + 1)
        ax1.axis("off")
        ax1.imshow(result[0].detach().numpy()[0][0], cmap="gray")
        ax1.set_title(torch.argmax(result[1]).item())
        ax2 = fig.add_subplot(2, num_seeds, num_seeds + s + 1)
        ax2.imshow(cv2.medianBlur(result[0].detach().numpy()[0][0], 5), cmap="gray")
        ax2.axis("off")
    plt.suptitle("Result of GS")
    plt.tight_layout()
    plt.savefig("gs.png")
