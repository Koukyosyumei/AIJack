import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from aijack.attack import DLG_Attack, GS_Attack
from aijack.utils import NumpyDataset

batch_size = 64


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


def prepare_dataloader(path="MNIST/."):
    at_t_dataset_train = torchvision.datasets.MNIST(
        root=path, train=True, download=True
    )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = NumpyDataset(
        at_t_dataset_train.train_data.numpy(),
        at_t_dataset_train.train_labels.numpy(),
        transform=transform,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    return dataloader


def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    net = Net().to(device)
    dataloader = prepare_dataloader()
    for data in dataloader:
        x, y = data[0], data[1]
        break

    criterion = nn.CrossEntropyLoss()
    pred = criterion(net(x[:1]), y[:1])
    client_gradients = torch.autograd.grad(pred, net.parameters())
    client_gradients = [cg.detach() for cg in client_gradients]

    attacker = DLG_Attack(net, (1, 1, 28, 28), (1, 10), criterion)
    result = attacker.attack(client_gradients, iteration=100, lr=0.1)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(result[1][0].detach().numpy()[0][0], cmap="gray")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(cv2.medianBlur(result[1][0].detach().numpy()[0][0], 5), cmap="gray")
    plt.suptitle(f"DLG - {torch.argmax(result[1][1]).item()}")
    plt.savefig("dlg.png")
    plt.close()

    attacker = GS_Attack(net, (1, 1, 28, 28), (1, 10), criterion, alpha=0.0001)
    result = attacker.attack(client_gradients, iteration=100, lr=0.05)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(result[1][0].detach().numpy()[0][0], cmap="gray")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(cv2.medianBlur(result[1][0].detach().numpy()[0][0], 5), cmap="gray")
    plt.suptitle(f"GS - {torch.argmax(result[1][1]).item()}")
    plt.savefig("gs.png")
    plt.close()


if __name__ == "__main__":
    main()
