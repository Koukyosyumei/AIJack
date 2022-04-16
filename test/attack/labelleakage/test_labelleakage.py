def test_labelleakage():
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler

    from aijack.attack import NormAttackManager
    from aijack.collaborative import SplitNN, SplitNNClient
    from aijack.utils import NumpyDataset

    torch.manual_seed(10)

    batch_size = 5
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

    train_df = pd.read_csv("test/demodata/demo_creditcard.csv")
    # You don't want the `Time` column.
    train_df.pop("Time")
    # The `Amount` column covers a huge range. Convert to log-space.
    eps = 0.001  # 0 => 0.1¢
    train_df["Log Ammount"] = np.log(train_df.pop("Amount") + eps)

    # Form np arrays of labels and features.
    train_labels = np.array(train_df.pop("Class"))

    train_features = np.array(train_df)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    train_features = np.clip(train_features, -5, 5)

    train_dataset = NumpyDataset(
        train_features, train_labels.astype(np.float64).reshape(-1, 1)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    model_1 = FirstNet(train_features)
    model_2 = SecondNet()
    model_1.double()
    model_2.double()
    opt_1 = optim.Adam(model_1.parameters(), lr=1e-3)
    opt_2 = optim.Adam(model_2.parameters(), lr=1e-3)
    optimizers = [opt_1, opt_2]
    client_1 = SplitNNClient(model_1, user_id=0)
    client_2 = SplitNNClient(model_2, user_id=0)
    clients = [client_1, client_2]
    criterion = nn.BCELoss()

    manager = NormAttackManager(criterion, device="cpu")
    NormAttackSplitNN = manager.attach(SplitNN)
    normattacksplitnn = NormAttackSplitNN(clients, optimizers)

    normattacksplitnn.train()
    loss_log = []
    for _ in range(2):
        for data in train_loader:
            inputs, labels = data
            normattacksplitnn.zero_grad()
            outputs = normattacksplitnn(inputs)
            loss = criterion(outputs, labels)
            normattacksplitnn.backward(loss)
            normattacksplitnn.step()
            loss_log.append(loss.item())

    train_leak_auc = normattacksplitnn.attack(train_loader)
    assert loss_log[0] > loss_log[1]
    assert train_leak_auc > 0.5
