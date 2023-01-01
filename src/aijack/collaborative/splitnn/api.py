from ..core.api import BaseFedAPI


class SplitNNAPI(BaseFedAPI):
    def __init__(self, clients, optimizers, dataloader, criterion, num_epoch):
        super().__init__()
        self.clients = clients
        self.optimizers = optimizers
        self.dataloader = dataloader
        self.criterion = criterion
        self.num_epoch = num_epoch

        self.num_clients = len(clients)
        self.recent_output = None
        self.loss_log = []

    def local_train(self):
        for data in self.dataloader:
            self.zero_grad()
            inputs, labels = data
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            self.backward(loss)
            self.step()

            self.loss_log.append(loss.item())

    def run(self):
        self.train()

        for _ in range(self.num_epoch):
            self.local_train()

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

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
