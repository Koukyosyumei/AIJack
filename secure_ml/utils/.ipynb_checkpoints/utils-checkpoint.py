import torch
from torch.utils.data.dataset import Dataset


def try_gpu(e):
    """send given tensor to gpu if it is available

    Args:
        e (torch.Tensor)

    Returns:
        e (torch.Tensor)
    """
    if torch.cuda.is_available():
        return e.cuda()
    return e


class DataSet(Dataset):
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
        """get the number of rows of self.x
        """
        return len(self.x)
