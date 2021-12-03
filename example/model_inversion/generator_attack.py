import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Subset

from aijack.attack import Generator_Attack
from aijack.collaborative import SplitNN, SplitNNClient


class FirstNet(nn.Module):
    def __init__(self):
        super(FirstNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1
        )
        # self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        # self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x):
        # 3ch > 64ch, shape 32 x 32 > 16 x 16
        x = self.conv1(x)  # [64,32,32]
        # x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)  # [64,16,16]

        # 64ch > 128ch, shape 16 x 16 > 8 x 8
        x = self.conv2(x)  # [128,16,16]
        # x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)  # [128,8,8]
        return x


# CNNを実装する
class SecondNet(nn.Module):
    def __init__(self):
        super(SecondNet, self).__init__()
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.L1 = nn.Linear(512, 10)  # 10クラス分類

    def forward(self, x):
        # 128ch > 256ch, shape 8 x 8 > 4 x 4
        x = self.conv3(x)  # [256,8,8]
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)  # [256,4,4]

        # 256ch > 512ch, shape 4 x 4 > 2 x 2
        x = self.conv4(x)  # [512,4,4]
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)  # [512,2,2]
        # 全結合層
        x = x.view(-1, 512)
        x = self.L1(x)
        # x = F.softmax(x, dim=0)
        return x


class Attacker(nn.Module):
    def __init__(self):
        super(Attacker, self).__init__()
        self.fla = nn.Flatten()
        self.ln1 = nn.Linear(128 * 7 * 7, 1000)
        self.ln2 = nn.Linear(1000, 784)

    def forward(self, x):
        x = self.fla(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.ln2(x)
        x = x.view(-1, 1, 28, 28)

        return x


def accuracy(label, output):
    pred = output.argmax(dim=1, keepdim=True)
    return pred.eq(label.view_as(pred)).sum().item() / pred.shape[0]


config = {"batch_size": 128}


def main():
    torch.random.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is ", device)

    transform = transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    victim_idx = random.sample(range(trainset.data.shape[0]), k=2000)
    victim_train_idx = victim_idx[:1000]
    attack_idx = victim_idx[1000:]
    victim_test_idx = random.sample(range(testset.data.shape[0]), k=15)
    victim_train_dataset = Subset(trainset, victim_train_idx)
    attack_dataset = Subset(trainset, attack_idx)
    victim_test_dataset = Subset(testset, victim_test_idx)
    victim_train_dataloader = torch.utils.data.DataLoader(
        victim_train_dataset, batch_size=64, shuffle=True
    )
    attack_dataloader = torch.utils.data.DataLoader(
        attack_dataset, batch_size=64, shuffle=True
    )
    victim_test_dataloader = torch.utils.data.DataLoader(
        victim_test_dataset, batch_size=64, shuffle=False
    )

    model_1 = FirstNet()
    model_1 = model_1.to(device)
    model_2 = SecondNet()
    model_2 = model_2.to(device)

    opt_1 = optim.Adam(model_1.parameters(), lr=1e-3)
    opt_2 = optim.Adam(model_2.parameters(), lr=1e-3)
    optimizers = [opt_1, opt_2]

    criterion = nn.CrossEntropyLoss()

    client_1 = SplitNNClient(model_1, user_id=0)
    client_2 = SplitNNClient(model_2, user_id=0)
    clients = [client_1, client_2]
    splitnn = SplitNN(clients)

    print("normal training")
    splitnn.train()
    for epoch in range(3):
        epoch_loss = 0
        epoch_outputs = []
        epoch_labels = []
        for i, data in enumerate(victim_train_dataloader):
            for opt in optimizers:
                opt.zero_grad()

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = splitnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            splitnn.backward(outputs.grad)
            epoch_loss += loss.item() / len(victim_train_dataloader.dataset)

            epoch_outputs.append(outputs)
            epoch_labels.append(labels)

            for opt in optimizers:
                opt.step()

        print(
            f"epoch={epoch}, loss: {epoch_loss}, accuracy: {accuracy(torch.cat(epoch_labels), torch.cat(epoch_outputs))}"
        )

    print("attacking...")
    attacker = Attacker()
    attacker = attacker.to(device)
    opt_3 = optim.Adam(attacker.parameters(), lr=1e-3)
    generator_attacker = Generator_Attack(splitnn.clients[0], attacker, opt_3)
    generator_attacker.fit(attack_dataloader, 10)
    result = generator_attacker.attack(victim_test_dataloader)
    print("reconstructing...")
    result = result.detach().numpy()

    for data, _ in victim_test_dataloader:
        break
    for i in range(1, 16):
        plt.subplot(3, 5, i)
        plt.imshow(data[i - 1].reshape(28, 28), cmap="gray_r")
    plt.suptitle("original images")
    plt.savefig("original.png")
    plt.close()

    for i in range(1, 16):
        plt.subplot(3, 5, i)
        plt.imshow(result[i - 1].reshape(28, 28), cmap="gray_r")
    plt.suptitle("reconstructed images")
    plt.savefig("reconstructed.png")

    print("done!")


if __name__ == "__main__":
    main()
