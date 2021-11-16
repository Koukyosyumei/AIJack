import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from aijack.attack import GAN_Attack
from aijack.collaborative import Client, Server
from aijack.utils import DataSet

# Number of channels in the training images. For color images this is 3
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Batch Size
batch_size = 64


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(Generator, self).__init__()
        # Generator Code (from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 4, 4, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 1, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.Tanh(),
            nn.MaxPool2d(3, 3, 1),
            nn.Conv2d(32, 64, 5),
            nn.Tanh(),
            nn.MaxPool2d(3, 3, 1),
        )

        self.lin = nn.Sequential(
            nn.Linear(256, 200), nn.Tanh(), nn.Linear(200, 11), nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape((-1, 256))
        x = self.lin(x)
        return x


def prepare_dataloaders():
    at_t_dataset_train = torchvision.datasets.MNIST(
        root="./", train=True, download=True
    )
    at_t_dataset_test = torchvision.datasets.MNIST(
        root="./", train=False, download=True
    )

    X = at_t_dataset_train.data.numpy()
    y = at_t_dataset_train.targets.numpy()

    # ToTensor：画像のグレースケール化（RGBの0~255を0~1の範囲に正規化）、Normalize：Z値化（RGBの平均と標準偏差を0.5で決め打ちして正規化）
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # idx_1 = random.sample(range(400), 200)
    # idx_2 = list(set(range(400)) - set(idx_1))
    idx_1 = np.where(y < 5)[0]
    idx_2 = np.where(y >= 5)[0]

    global_trainset = DataSet(
        at_t_dataset_test.data.numpy(),
        at_t_dataset_test.targets.numpy(),
        transform=transform,
    )
    global_trainloader = torch.utils.data.DataLoader(
        global_trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    trainset_1 = DataSet(X[idx_1], y[idx_1], transform=transform)
    trainloader_1 = torch.utils.data.DataLoader(
        trainset_1, batch_size=batch_size, shuffle=True, num_workers=2
    )
    trainset_2 = DataSet(X[idx_2], y[idx_2], transform=transform)
    trainloader_2 = torch.utils.data.DataLoader(
        trainset_2, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return X, y, [trainloader_1, trainloader_2], global_trainloader, [200, 200]


def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    print(device)

    X, y, trainloaders, global_trainloader, dataset_nums = prepare_dataloaders()

    criterion = nn.CrossEntropyLoss()
    client_num = 2
    adversary_client_id = 1
    target_label = 3

    net_1 = Net()
    client_1 = Client(net_1, user_id=0)
    client_1.to(device)
    optimizer_1 = optim.SGD(
        client_1.parameters(), lr=0.02, weight_decay=1e-7, momentum=0.9
    )

    net_2 = Net()
    client_2 = Client(net_2, user_id=1)
    client_2.to(device)
    optimizer_2 = optim.SGD(
        client_2.parameters(), lr=0.02, weight_decay=1e-7, momentum=0.9
    )

    clients = [client_1, client_2]
    optimizers = [optimizer_1, optimizer_2]

    generator = Generator(nz, nc, ngf)
    generator.to(device)
    optimizer_g = optim.SGD(
        generator.parameters(), lr=0.05, weight_decay=1e-7, momentum=0.0
    )
    gan_attacker = GAN_Attack(
        client_2,
        target_label,
        generator,
        optimizer_g,
        criterion,
        nz=nz,
        device=device,
    )

    global_model = Net()
    global_model.to(device)
    server = Server(clients, global_model)

    fake_batch_size = batch_size
    fake_label = 10

    for epoch in range(10):
        for client_idx in range(client_num):
            client = clients[client_idx]
            trainloader = trainloaders[client_idx]
            optimizer = optimizers[client_idx]

            running_loss = 0.0
            for _, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                if epoch != 0 and client_idx == adversary_client_id:
                    fake_image = gan_attacker.attack(fake_batch_size)
                    inputs = torch.cat([inputs, fake_image])
                    labels = torch.cat(
                        [
                            labels,
                            torch.tensor([fake_label] * fake_batch_size, device=device),
                        ]
                    )

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = client(inputs)
                loss = criterion(outputs, labels.to(torch.int64))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(
                f"epoch {epoch}: client-{client_idx+1}",
                running_loss / dataset_nums[client_idx],
            )

        server.update()
        server.distribtue()

        gan_attacker.update_discriminator()
        gan_attacker.update_generator(batch_size=64, epoch=1000, log_interval=100)

        in_preds = []
        in_label = []
        with torch.no_grad():
            for data in global_trainloader:
                inputs, labels = data
                inputs = inputs.to(device)
                outputs = server.global_model(inputs)
                in_preds.append(outputs)
                in_label.append(labels)
            in_preds = torch.cat(in_preds)
            in_label = torch.cat(in_label)
        print(
            f"epoch {epoch}: accuracy is ",
            accuracy_score(
                np.array(torch.argmax(in_preds, axis=1).cpu()), np.array(in_label)
            ),
        )

        reconstructed_image = gan_attacker.attack(1).cpu().numpy().reshape(28, 28)
        print(
            "reconstrunction error is ",
            np.sqrt(
                np.sum(
                    (
                        (X[np.where(y == target_label)[0][:10], :, :] - 0.5 / 0.5)
                        - reconstructed_image
                    )
                    ** 2
                )
            )
            / (10 * (28 * 28)),
        )
        plt.imshow(reconstructed_image * 0.5 + 0.5, vmin=-1, vmax=1)
        plt.savefig(f"{epoch}.png")


if __name__ == "__main__":
    main()
