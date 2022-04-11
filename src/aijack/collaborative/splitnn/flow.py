import torch


class SplitNN(torch.nn.Module):
    def __init__(self, clients, optimizers):
        super().__init__()
        self.clients = clients
        self.optimizers = optimizers
        self.num_clients = len(clients)
        self.recent_output = None

    def forward(self, x):
        intermidiate_to_next_client = x
        for client in self.clients:
            intermidiate_to_next_client = client.upload(intermidiate_to_next_client)
        output = intermidiate_to_next_client
        self.recent_output = output
        return output

    def backward(self, loss):
        loss.backward()
        return self.backward_gradient(self.recent_output.grad)

    def backward_gradient(self, grads_outputs):
        grad_from_next_client = grads_outputs
        for i in range(self.num_clients - 1, -1, -1):
            self.clients[i].download(grad_from_next_client)
            if i != 0:
                grad_from_next_client = self.clients[i].distribute()
        return grad_from_next_client

    def train(self):
        for client in self.clients:
            client.train()

    def eval(self):
        for client in self.clients:
            client.train()

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()
