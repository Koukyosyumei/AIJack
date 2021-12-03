import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from aijack.attack import SplitNNNormAttack
from aijack.collaborative import SplitNN, SplitNNClient
from aijack.utils import DataSet

config = {"batch_size": 128}

hidden_dim = 16


class FirstNet(nn.Module):
    def __init__(self, train_features):
        super(FirstNet, self).__init__()
        self.L1 = nn.Linear(train_features.shape[-1], hidden_dim)

    def forward(self, x):
        x = self.L1(x)
        x = nn.functional.relu(x)
        return x


class SecondNet(nn.Module):
    def __init__(self):
        super(SecondNet, self).__init__()
        self.L2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.L2(x)
        x = torch.sigmoid(x)
        return x


def torch_auc(label, pred):
    return roc_auc_score(label.detach().numpy(), pred.detach().numpy())


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is ", device)
    raw_df = pd.read_csv(
        "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    )
    raw_df_neg = raw_df[raw_df["Class"] == 0]
    raw_df_pos = raw_df[raw_df["Class"] == 1]

    down_df_neg = raw_df_neg  # .sample(40000)
    down_df = pd.concat([down_df_neg, raw_df_pos])

    neg, pos = np.bincount(down_df["Class"])
    total = neg + pos
    print(
        "Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(
            total, pos, 100 * pos / total
        )
    )

    cleaned_df = down_df.copy()
    # You don't want the `Time` column.
    cleaned_df.pop("Time")
    # The `Amount` column covers a huge range. Convert to log-space.
    eps = 0.001  # 0 => 0.1¢
    cleaned_df["Log Ammount"] = np.log(cleaned_df.pop("Amount") + eps)

    # Use a utility from sklearn to split and shuffle our dataset.
    train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)

    # Form np arrays of labels and features.
    train_labels = np.array(train_df.pop("Class"))
    val_labels = np.array(val_df.pop("Class"))
    test_labels = np.array(test_df.pop("Class"))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)
    train_features = np.clip(train_features, -5, 5)
    val_features = np.clip(val_features, -5, 5)
    test_features = np.clip(test_features, -5, 5)
    print("Training labels shape:", train_labels.shape)
    print("Validation labels shape:", val_labels.shape)
    print("Test labels shape:", test_labels.shape)
    print("Training features shape:", train_features.shape)
    print("Validation features shape:", val_features.shape)
    print("Test features shape:", test_features.shape)

    train_dataset = DataSet(
        train_features, train_labels.astype(np.float64).reshape(-1, 1)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_dataset = DataSet(test_features, test_labels.astype(np.float64).reshape(-1, 1))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=True
    )

    model_1 = FirstNet(train_features)
    model_1 = model_1.to(device)
    model_2 = SecondNet()
    model_2 = model_2.to(device)
    model_1.double()
    model_2.double()
    opt_1 = optim.Adam(model_1.parameters(), lr=1e-3)
    opt_2 = optim.Adam(model_2.parameters(), lr=1e-3)
    optimizers = [opt_1, opt_2]
    criterion = nn.BCELoss()
    client_1 = SplitNNClient(model_1, user_id=0)
    client_2 = SplitNNClient(model_2, user_id=0)
    clients = [client_1, client_2]
    splitnn = SplitNN(clients)

    splitnn.train()
    for epoch in range(3):
        epoch_loss = 0
        epoch_outputs = []
        epoch_labels = []
        for i, data in enumerate(train_loader):
            for opt in optimizers:
                opt.zero_grad()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = splitnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            splitnn.backward(outputs.grad)
            epoch_loss += loss.item() / len(train_loader.dataset)

            epoch_outputs.append(outputs)
            epoch_labels.append(labels)

            for opt in optimizers:
                opt.step()

        print(
            f"epoch={epoch}, loss: {epoch_loss}, auc: {torch_auc(torch.cat(epoch_labels), torch.cat(epoch_outputs))}"
        )

    nall = SplitNNNormAttack(splitnn)
    train_leak_auc = nall.attack(train_loader, criterion, device)
    print("Leau AUC is ", train_leak_auc)
    test_leak_auc = nall.attack(test_loader, criterion, device)
    print("Leau AUC is ", test_leak_auc)


if __name__ == "__main__":
    main()
