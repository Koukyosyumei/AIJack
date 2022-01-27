import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ...utils import NumpyDataset, try_gpu
from ..base_attack import BaseAttacker


class ShadowModel:
    """Train shadow models for membership inference
       reference https://arxiv.org/abs/1610.05820

    Args
        models : torch models for shadow
        shadow_dataset_suze (int) : size of dataset for shadow models
        shadow_transfomrm (torch.transform) : transform
        seed (int) : random seed

    Attriutes
        models : torch models for shadow
        shadow_dataset_suze (int) : size of dataset for shadow models
        shadow_transfomrm (torch.transform) : transform
        seed (int) : random seed
        num_models(int):
        trainloaders
        testloaders

    """

    def __init__(
        self,
        models,
        shadow_dataset_size,
        shadow_transform=None,
        seed=42,
    ):

        self.models = models
        self.shadow_dataset_size = shadow_dataset_size
        self.shadow_transform = shadow_transform
        self.seed = seed

        self.num_models = len(models)
        self.trainloaders = []
        self.testloaders = []

        # initialize random state
        self._reset_random_state()

    def _reset_random_state(self):
        """initialize random state"""
        self._prng = np.random.RandomState(self.seed)

    def fit_transform(self, X, y, num_itr=100):
        """train shadow models and get prediction of shadow models
           and membership label of each prediction for each class

        Args:
            X (np.array): target data
            y (np.array): target label
            num_itr (int): number of iteration for training

        Returns:
            result_dict (dict) : key is each class
                                 value is (shadow_data, shadow_label)
        """
        self._fit(X, y, num_itr=num_itr)
        (
            shadow_in_data,
            shadow_out_data,
            in_original_labels,
            out_original_labels,
        ) = self._transform()

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
                shadow_data[local_idx],
                shadow_label[local_idx],
            )

        return result_dict

    def _fit(self, X, y, num_itr=100):
        """train shadow models on given data

        Args:
            X (np.array): training data for shadow models
            y (np.array): training label for shadow models
            num_itr (int): number of iteration for training
        """

        indices = np.arange(X.shape[0])

        for model_idx in range(self.num_models):
            model = self.models[model_idx]

            shadow_indices = self._prng.choice(
                indices, 2 * self.shadow_dataset_size, replace=False
            )

            train_indices = shadow_indices[: self.shadow_dataset_size]
            test_indices = shadow_indices[self.shadow_dataset_size :]

            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            trainset = NumpyDataset(X_train, y_train, transform=self.shadow_transform)
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=4, shuffle=True, num_workers=2
            )
            testset = NumpyDataset(X_test, y_test, transform=self.shadow_transform)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=4, shuffle=True, num_workers=2
            )

            self.trainloaders.append(trainloader)
            self.testloaders.append(testloader)

            # training
            # TODO : allow users to use specified loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

            for epoch in range(num_itr):
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    inputs = try_gpu(inputs)
                    labels = try_gpu(labels)
                    optimizer.zero_grad()
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    if i % 2000 == 1999:
                        print(
                            "[%d, %5d] loss: %.3f"
                            % (epoch + 1, i + 1, running_loss / 2000)
                        )
                        running_loss = 0.0
            print("Finished Training")

            self.models[model_idx] = model

    def _transform(self):
        """get prediction and its membership label per each class
           from shadow models

        Returns:
            shadow_in_data (torch.Tensor): prediction from shadow model
                                           on in_data
                                           (in_data means training data of
                                            each shadow model)
            shadow_out_data (torch.Tensor): prediction from shadow model
                                            on out_data
                                           (out_data means the data which
                                            each shadow model has not seen)
            in_original_labels (torch.Tensor): membership labels for
                                               prediciton on in_data
                                               (they should be all positive)
            out_original_labels (torch.Tensor): membership labels for
                                                prediciton on out__data
                                               (they should be all negative)
        """

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
                    inputs = try_gpu(inputs)
                    labels = try_gpu(labels)
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
                    inputs = try_gpu(inputs)
                    labels = try_gpu(labels)
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

        return shadow_in_data, shadow_out_data, in_original_labels, out_original_labels

    def get_score(self):
        pass


class AttackerModel:
    def __init__(self, models):
        """train attack model for memnership inference
           reference https://arxiv.org/abs/1610.05820

        Args:
            models: models of attacker

        Attriutes:
            _init_models
            models
        """
        self._init_models = models
        self.models = {}

    def fit(self, shadow_result):
        """train an attacl model with the result of shadow models

        Args:
            shadow_result (dict): key is each class
                                  value is (shadow_data, shadow_label)
        """
        for model_idx, (label, (X, y)) in enumerate(shadow_result.items()):
            model = self._init_models[model_idx]
            X = np.array(X.cpu())
            y = np.array(y.cpu())
            model.fit(X, y)
            self.models[label] = model

    def predict(self, y_pred_prob, y_labels):
        """predict whether the given prediction came from training data or not

        Args:
            y_pred_prob (torch.Tensor): predicted probabilities on the data
            y_labels (torch.Tensor): true label of the data which y_pred_prob
                                     is predicted on

        Returns:
            in_out_pred (np.array) : result of attack
                                     each element should be one or zero
        """
        y_pred_prob = np.array(y_pred_prob.cpu())
        y_labels = np.array(y_labels.cpu())
        unique_labels = np.unique(y_labels)
        in_out_pred = np.zeros_like(y_labels)
        for label in unique_labels:
            idx = np.where(y_labels == label)
            in_out_pred[idx] = self.models[label].predict(y_pred_prob[idx])

        return in_out_pred

    def predict_proba(self, y_pred_prob, y_labels):
        """get probabilities of whether the given prediction came from
           training data or not

        Args:
            y_pred_prob (torch.Tensor): predicted probabilities on the data
            y_labels (torch.Tensor): true label of the data which y_pred_prob
                                     is predicted on

        Returns:
            in_out_pred (np.array) : result of attack
                                     each element expresses the possibility
        """
        y_pred_prob = np.array(y_pred_prob.cpu())
        y_labels = np.array(y_labels.cpu())
        unique_labels = np.unique(y_labels)
        in_out_pred = np.zeros_like(y_labels).astype(float)
        for label in unique_labels:
            idx = np.where(y_labels == label)
            in_out_pred[idx] = self.models[label].predict_proba(y_pred_prob[idx])[:, 1]

        return in_out_pred


class Membership_Inference(BaseAttacker):
    def __init__(
        self,
        target_model,
        shadow_models,
        attack_models,
        shadow_data_size,
        shadow_transform,
    ):
        """Implementation of membership inference
           reference https://arxiv.org/abs/1610.05820

        Args:
            target_model: the model of the victim
            shadow_models: shadow model for attack
            attack_models: attacker model for attack
            shadow_data_size: the size of datasets for
                              training the shadow models
            shadow_transform: the transformation function for shadow datasets

        Attributes:
            shadow_models
            attack_models
            shadow_data_size
            shadow_trnasform
            sm
            shadow_result
            am
        """
        super().__init__(target_model)
        self.shadow_models = shadow_models
        self.attack_models = attack_models
        self.shadow_data_size = shadow_data_size
        self.shadow_trasform = shadow_transform

        self.sm = None
        self.shadow_result = None
        self.am = None

    def train_shadow(self, X, y, num_itr):
        """train shadow models

        Args:
            X (np.array): training data for shadow models
            y (np.array): training label for shadow models
            num_itr (int): number of iteration for training
        """
        self.sm = ShadowModel(
            self.shadow_models,
            self.shadow_data_size,
            shadow_transform=self.shadow_trasform,
        )
        self.shadow_result = self.sm.fit_transform(X, y, num_itr=num_itr)

    def train_attacker(self):
        """Train attacker models"""
        self.am = AttackerModel(self.attack_models)
        self.am.fit(self.shadow_result)

    def attack(self, x, y, proba=False):
        """Attack victim model

        Args:
            x: target datasets which the attacker wants to classify
            y: target labels which the attacker wants to classify
            proba: the format of the output
        """
        prediction_of_taregt_model = self.target_model(x)
        if proba:
            return self.predit_proba(prediction_of_taregt_model, y)
        else:
            return self.predit(prediction_of_taregt_model, y)

    def predict(self, pred, label):
        """Predict whether the given prediction came from training data or not

        Args:
            pred (torch.Tensor): predicted probabilities on the data
            label (torch.Tensor): true label of the data which y_pred_prob
                                     is predicted on

        Returns:
            predicted binaru labels
        """
        return self.am.predict(pred, label)

    def predict_proba(self, pred, label):
        """get probabilities of whether the given prediction came from
           training data or not

        Args:
            pred (torch.Tensor): predicted probabilities on the data
            label (torch.Tensor): true label of the data which y_pred_prob
                                     is predicted on

        Returns:
            predicted probabilities
        """
        return self.am.predict_proba(pred, label)
