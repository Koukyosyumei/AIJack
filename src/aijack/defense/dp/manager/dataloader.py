import numpy as np
import torch
from torch.utils.data import DataLoader


class PoissonSampler:
    def __init__(self, dataset, lot_size, iterations):
        self.dataset_size = len(dataset)
        self.lot_size = lot_size
        self.iterations = iterations

    def __iter__(self):
        for _ in range(self.iterations):
            indices = np.where(
                torch.rand(self.dataset_size) < (self.lot_size / self.dataset_size)
            )[0]
            if indices.size > 0:
                yield indices

    def __len__(self):
        return self.iterations


class DPWrapperLotDataIterator:
    def __init__(self, original_iterator, dp_optimizer):
        self.original_iterator = original_iterator
        self.dp_optimizer = dp_optimizer
        self.init_flag = True

    def __iter__(self):
        return self

    def _reset(self, *args, **kwargs):
        return self.original_iterator._reset(*args, **kwargs)

    def _next_index(self, *args, **kwargs):
        return self.original_iterator._next_index(*args, **kwargs)

    def __next__(self):
        if not self.init_flag:
            self.dp_optimizer.step_for_lot()
        else:
            self.init_flag = False

        data = self.original_iterator.__next__()
        self.dp_optimizer.zero_grad_for_lot()

        return data

    def __len__(self):
        return self.original_iterator.__len__()

    def __getstate__(self):
        raise self.original_iterator.__getstate__()


class LotDataLoader(DataLoader):
    def __init__(self, dp_optimizer, *args, **kwargs):
        super(LotDataLoader, self).__init__(*args, **kwargs)
        self.dp_optimizer = dp_optimizer
        self.init_flag = True

    def __iter__(self):
        return DPWrapperLotDataIterator(
            super(LotDataLoader, self).__iter__(), self.dp_optimizer
        )
