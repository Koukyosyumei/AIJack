import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from aijack.attack import MI_FACE
from aijack.utils import DataSet
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

# INPUT PATHS:
BASE = "data/"


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


def main():
    imgs = []
    labels = []
    for i in range(1, 41):
        for j in range(1, 11):
            img = cv2.imread(BASE + f"s{i}/{j}.pgm", 0)
            imgs.append(img)
            labels.append(i - 1)

    X = np.stack(imgs)
    y = np.array(labels)

    # ToTensor：画像のグレースケール化（RGBの0~255を0~1の範囲に正規化）、Normalize：Z値化（RGBの平均と標準偏差を0.5で決め打ちして正規化）
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = DataSet(X, y, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0
        data_size = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.to(torch.int64))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            data_size += inputs.shape[0]

        print(f"epoch {epoch}: loss is {running_loss/data_size}")

    print("Finished Training")

    in_preds = []
    in_label = []
    with torch.no_grad():
        for data in trainloader:
            inputs, labels = data
            outputs = net(inputs)
            in_preds.append(outputs)
            in_label.append(labels)
        in_preds = torch.cat(in_preds)
        in_label = torch.cat(in_label)
    print(accuracy_score(np.array(torch.argmax(in_preds, axis=1)), np.array(in_label)))

    input_shape = (1, 1, 112, 92)
    target_label_1 = 1
    target_label_2 = 10
    lam = 0.1
    num_itr = 100
    print("start model inversion")
    mi = MI_FACE(net, input_shape)
    print("finish model inversion")
    print("reconstruct images ....")
    x_result_1, log = mi.attack(target_label_1, lam, num_itr)
    x_result_2, log = mi.attack(target_label_2, lam, num_itr)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 5))
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
    main()
