import numpy as np
import torch


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
