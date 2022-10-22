import random
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mpi4py import MPI
from torchvision import datasets, transforms

from aijack.collaborative.fedavg import MPIFedAVGAPI, MPIFedAVGClient, MPIFedAVGServer

logger = getLogger(__name__)

training_batch_size = 64
test_batch_size = 64
num_rounds = 5
lr = 0.05
seed = 0


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def prepare_dataloader(num_clients, myid, train=True, path=""):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    if train:
        dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
        idxs = list(range(len(dataset.data)))
        random.shuffle(idxs)
        idx = np.array_split(idxs, num_clients, 0)[myid - 1]
        dataset.data = dataset.data[idx]
        dataset.targets = dataset.targets[idx]
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=training_batch_size
        )
        return train_loader
    else:
        dataset = datasets.MNIST(path, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size)
        return test_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.ln = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = self.ln(x.reshape(-1, 28 * 28))
        output = F.log_softmax(x, dim=1)
        return output


def evaluate_gloal_model(dataloader):
    def _evaluate_global_model(api):
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(api.party.device), target.to(api.party.device)
                output = api.party.server_model(data)
                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(dataloader.dataset)
        accuracy = 100.0 * correct / len(dataloader.dataset)
        print(
            f"Round: {api.party.round}, Test set: Average loss: {test_loss}, Accuracy: {accuracy}"
        )

    return _evaluate_global_model


def main():
    fix_seed(seed)

    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    size = comm.Get_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    if myid == 0:
        dataloader = prepare_dataloader(size - 1, myid, train=False)
        client_ids = list(range(1, size))
        server = MPIFedAVGServer(comm, model, myid, client_ids, myid, lr, "sgd")
        api = MPIFedAVGAPI(
            comm,
            server,
            True,
            F.nll_loss,
            None,
            None,
            num_rounds,
            1,
            custom_action=evaluate_gloal_model(dataloader),
        )
    else:
        dataloader = prepare_dataloader(size - 1, myid, train=True)
        client = MPIFedAVGClient(comm, model, myid, lr, device=device)
        api = MPIFedAVGAPI(
            comm,
            client,
            False,
            F.nll_loss,
            optimizer,
            dataloader,
            num_rounds,
            1,
        )

    t1 = MPI.Wtime()
    api.run()
    t2 = MPI.Wtime()

    t0 = np.ndarray(1, dtype="float64")
    t_w = np.ndarray(1, dtype="float64")
    t0[0] = t2 - t1
    comm.Reduce(t0, t_w, op=MPI.MAX, root=0)
    if myid == 0:
        print("Execution time = : ", t_w[0], "  [sec.] \n")


if __name__ == "__main__":
    main()
