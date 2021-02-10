import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset


class DataSet(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.x)

# Shadow model


class ShadowModel:
    def __init__(self, models,
                 shadow_dataset_size,
                 shadow_transform=None,
                 seed=42,
                 ):

        self.models = models
        self.shadow_dataset_size = shadow_dataset_size
        self.num_models = len(models)
        self.seed = seed
        self.shadow_transform = shadow_transform

        self._reset_random_state()
        self.trainloaders = []
        self.testloaders = []

    def _reset_random_state(self):
        self._prng = np.random.RandomState(self.seed)

    def fit_transform(self, X, y, num_itr=100):
        self._fit(X, y, num_itr=num_itr)
        shadow_in_data, shadow_out_data,\
            in_original_labels, out_original_labels = self._transform()

        shadow_in_label = torch.ones_like(in_original_labels)
        shadow_out_label = torch.zeros_like(out_original_labels)
        shadow_data = torch.cat([shadow_in_data, shadow_out_data])
        shadow_label = torch.cat([shadow_in_label, shadow_out_label])
        original_labels = torch.cat([in_original_labels, out_original_labels])

        result_dict = {}
        unique_labels = torch.unique(original_labels)
        for label in unique_labels:
            local_idx = torch.where(original_labels == label)
            result_dict[label.item()] = (
                shadow_data[local_idx], shadow_label[local_idx])

        return result_dict

    def _fit(self, X, y, num_itr=100):

        indices = np.arange(X.shape[0])

        for model_idx in range(self.num_models):
            model = self.models[model_idx]

            shadow_indices = self._prng.choice(
                indices, 2 * self.shadow_dataset_size, replace=False
            )

            train_indices = shadow_indices[: self.shadow_dataset_size]
            test_indices = shadow_indices[self.shadow_dataset_size:]

            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            trainset = DataSet(
                X_train, y_train, transform=self.shadow_transform)
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=4, shuffle=True, num_workers=2)
            testset = DataSet(X_test, y_test, transform=self.shadow_transform)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=4, shuffle=True, num_workers=2)

            self.trainloaders.append(trainloader)
            self.testloaders.append(testloader)

            # training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

            for epoch in range(num_itr):
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    if i % 2000 == 1999:
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 2000))
                        running_loss = 0.0
            print('Finished Training')

            self.models[model_idx] = model

    def _transform(self):

        shadow_in_data = []
        shadow_out_data = []
        in_original_labels = []
        out_original_labels = []

        for model_idx in range(self.num_models):
            model = self.models[model_idx]
            trainloader = self.trainloaders[model_idx]
            testloader = self.testloaders[model_idx]

            # shadow-in
            train_preds = []
            train_label = []
            with torch.no_grad():
                for data in trainloader:
                    inputs, labels = data
                    outputs = model(inputs)
                    train_preds.append(outputs)
                    train_label.append(labels)
            train_preds = torch.cat(train_preds)
            shadow_in_data.append(train_preds)
            train_label = torch.cat(train_label)
            in_original_labels.append(train_label)

            # shadow-out
            test_preds = []
            test_label = []
            with torch.no_grad():
                for data in testloader:
                    inputs, labels = data
                    outputs = model(inputs)
                    test_preds.append(outputs)
                    test_label.append(labels)
            test_preds = torch.cat(test_preds)
            shadow_out_data.append(test_preds)
            test_label = torch.cat(test_label)
            out_original_labels.append(test_label)

        shadow_in_data = torch.cat(shadow_in_data)
        shadow_out_data = torch.cat(shadow_out_data)
        in_original_labels = torch.cat(in_original_labels)
        out_original_labels = torch.cat(out_original_labels)

        return shadow_in_data, shadow_out_data,\
            in_original_labels, out_original_labels

    def get_score(self):
        pass


class AttackerModel:
    def __init__(self, models):
        self._init_models = models
        self.models = {}

    def fit(self, shadow_result):
        for model_idx, (label, (X, y)) in enumerate(shadow_result.items()):
            model = self._init_models[model_idx]
            X = np.array(X)
            y = np.array(y)
            model.fit(X, y)
            self.models[label] = model

    def predict(self, y_pred_prob, y_labels):
        y_pred_prob = np.array(y_pred_prob)
        y_labels = np.array(y_labels)
        unique_labels = np.unique(y_labels)
        in_out_pred = np.zeros_like(y_labels)
        for label in unique_labels:
            idx = np.where(y_labels == label)
            in_out_pred[idx] = self.models[label].predict(y_pred_prob[idx])

        return in_out_pred
