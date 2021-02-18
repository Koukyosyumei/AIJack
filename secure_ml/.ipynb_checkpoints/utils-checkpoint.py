import torch
from torch.utils.data.dataset import Dataset


def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e


class DataSet(Dataset):
    """
    This class allows you to convert numpy.array to torch.Dataset
    """

    def __init__(self, x, y, transform=None):
        """
        Attriutes
            x (np.array) :
            y (np.array) :
            transform (torch.transform)
        """
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
