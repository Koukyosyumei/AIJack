import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset

from aijack.attack import MI_FACE
from aijack.defense import GeneralMomentAccountant, PrivacyManager
from aijack.utils import DataSet

# INPUT PATHS:
BASE = "data/"

lot_size = 40
batch_size = 1
iterations = 10
sigma = 0.5
l2_norm_clip = 1
delta = 1e-5


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fla = nn.Flatten()
        self.fc = nn.Linear(112 * 92, 40)

    def forward(self, x):
        x = self.fla(x)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x


def prepare_dataset():
    imgs = []
    labels = []
    for i in range(1, 41):
        for j in range(1, 11):
            img = cv2.imread(BASE + f"s{i}/{j}.pgm", 0)
            imgs.append(img)
            labels.append(i - 1)

    X = np.stack(imgs)
    y = np.array(labels)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = DataSet(X, y, transform=transform)
    return trainset


def train():
    trainset = prepare_dataset()
    accountant = GeneralMomentAccountant(
        noise_type="Gaussian",
        search="ternary",
        precision=0.001,
        order_max=1,
        order_min=72,
        max_iterations=1000,
        bound_type="rdp_upperbound_closedformula",
        backend="python",
    )

    privacy_manager = PrivacyManager(
        accountant,
        optim.SGD,
        l2_norm_clip=l2_norm_clip,
        dataset=trainset,
        lot_size=lot_size,
        batch_size=batch_size,
        iterations=iterations,
    )

    accountant.reset_step_info()
    accountant.add_step_info(
        {"sigma": sigma},
        lot_size / len(trainset),
        iterations * (len(trainset) / lot_size),
    )
    estimated_epsilon = accountant.get_epsilon(delta=delta)
    print(f"estimated epsilon is {estimated_epsilon}")

    accountant.reset_step_info()
    dpoptimizer_cls, lot_loader, batch_loader = privacy_manager.privatize(
        noise_multiplier=sigma
    )

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = dpoptimizer_cls(net.parameters(), lr=0.05, momentum=0.9)

    for epoch in range(iterations):  # loop over the dataset multiple times
        running_loss = 0
        data_size = 0
        preds = []
        labels = []
        for data in lot_loader(trainset):
            X_lot, y_lot = data
            optimizer.zero_grad()
            for X_batch, y_batch in batch_loader(TensorDataset(X_lot, y_lot)):
                optimizer.zero_grad_keep_accum_grads()

                pred = net(X_batch)
                loss = criterion(pred, y_batch.to(torch.int64))
                loss.backward()
                optimizer.update_accum_grads()

                running_loss += loss.item()
                data_size += X_batch.shape[0]
                preds.append(pred)
                labels.append(y_batch)

            optimizer.step()

        preds = torch.cat(preds)
        labels = torch.cat(labels)
        print(f"epoch {epoch}: loss is {running_loss/data_size}")
        print(
            f"epoch {epoch}: accuracy is {accuracy_score(np.array(torch.argmax(preds, axis=1)), np.array(labels))}"
        )

    print(f"final epsilon is {accountant.get_epsilon(delta=delta)}")

    return net


def attack(net):
    input_shape = (1, 1, 112, 92)
    target_label_1 = 1
    target_label_2 = 10
    lam = 0.1
    num_itr = 100
    print("start model inversion")
    mi = MI_FACE(net, input_shape)
    print("finish model inversion")
    print("reconstruct images ....")
    x_result_1, _ = mi.attack(target_label_1, lam, num_itr)
    x_result_2, _ = mi.attack(target_label_2, lam, num_itr)

    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 5))
    axes[0][0].imshow(cv2.imread(BASE + "s2/1.pgm", 0), cmap="gray")
    axes[0][0].axis("off")
    axes[0][0].set_title("original image")
    axes[0][1].imshow(x_result_1[0][0], cmap="gray")
    axes[0][1].axis("off")
    axes[0][1].set_title("extracted image")

    axes[1][0].imshow(cv2.imread(BASE + "s11/1.pgm", 0), cmap="gray")
    axes[1][0].axis("off")
    axes[1][0].set_title("original image")
    axes[1][1].imshow(x_result_2[0][0], cmap="gray")
    axes[1][1].axis("off")
    axes[1][1].set_title("extracted image")
    plt.savefig("reconstructed.png")


if __name__ == "__main__":
    model = train()
    attack(model)
