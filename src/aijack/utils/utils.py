import random

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from torch.utils.data.dataset import Dataset


def default_local_train_for_client(
    self, local_epoch, criterion, trainloader, optimizer
):
    running_loss = 0.0
    for _ in range(local_epoch):
        for data in trainloader:
            _, x, y = data
            x = x.to(self.device)
            y = y.to(self.device).to(torch.int64)

            optimizer.zero_grad()
            loss = criterion(self(x), y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    return running_loss


def try_gpu(e):
    """Send given tensor to gpu if it is available

    Args:
        e: (torch.Tensor)

    Returns:
        e: (torch.Tensor)
    """
    if torch.cuda.is_available():
        return e.cuda()
    return e


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class RoundDecimal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_digits):
        ctx.save_for_backward(input)
        ctx.n_digits = n_digits
        return torch.round(input * 10**n_digits) / (10**n_digits)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return torch.round(grad_input * 10**ctx.n_digits) / (10**ctx.n_digits), None


torch_round_x_decimal = RoundDecimal.apply


class NumpyDataset(Dataset):
    """This class allows you to convert numpy.array to torch.Dataset

    Args:
        x (np.array):
        y (np.array):
        transform (torch.transform):

    Attriutes
        x (np.array):
        y (np.array):
        transform (torch.transform):
    """

    def __init__(self, x, y=None, transform=None, return_idx=False):
        self.x = x
        self.y = y
        self.transform = transform
        self.return_idx = return_idx

    def __getitem__(self, index):
        x = self.x[index]
        if self.y is not None:
            y = self.y[index]

        if self.transform is not None:
            x = self.transform(x)

        if not self.return_idx:
            if self.y is not None:
                return x, y
            else:
                return x
        else:
            if self.y is not None:
                return index, x, y
            else:
                return index, x

    def __len__(self):
        """get the number of rows of self.x"""
        return len(self.x)


class TorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        epoch=1,
        device="cpu",
        batch_size=1,
        shuffle=True,
        num_workers=2,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = epoch
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def fit(self, X, y):
        dataloader = torch.utils.data.DataLoader(
            NumpyDataset(X, y),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
        self.model.train()
        for _ in range(self.epoch):
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(x_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

        return self

    def predict_proba(self, X):
        dataloader = torch.utils.data.DataLoader(
            NumpyDataset(X),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.model.eval()
        y_pred_list = []
        with torch.no_grad():
            for x_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_pred = self.model(x_batch)
                y_pred_list.append(y_pred)
        return torch.cat(y_pred_list).cpu().detach().numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return accuracy_score(self.predict(X), y)
