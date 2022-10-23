import copy

import torch

from ..fedavg import FedAVGAPI, MPIFedAVGAPI


class FedProxAPI(FedAVGAPI):
    """Implementation of FedProx (https://arxiv.org/abs/1812.06127)"""

    def __init__(self, *args, mu=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu

    def run(self):
        for com in range(self.num_communication):
            for client_idx in range(self.client_num):
                client = self.clients[client_idx]
                trainloader = self.local_dataloaders[client_idx]
                optimizer = self.local_optimizers[client_idx]

                for i in range(self.local_epoch):
                    running_loss = 0.0
                    running_data_num = 0
                    for _, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        inputs = inputs.to(self.device)
                        inputs.requires_grad = True
                        labels = labels.to(self.device)

                        optimizer.zero_grad()
                        client.zero_grad()

                        outputs = client(inputs)
                        loss = self.criterion(outputs, labels)
                        loss.backward()

                        for local_param, global_param in zip(
                            client.parameters(), self.server.parameters()
                        ):
                            loss += (
                                self.mu
                                / 2
                                * torch.norm(local_param.data - global_param.data)
                            )
                            local_param.grad.data += self.mu * (
                                local_param.data - global_param.data
                            )

                        optimizer.step()

                        running_loss += loss.item()
                        running_data_num += inputs.shape[0]

                    print(
                        f"communication {com}, epoch {i}: client-{client_idx+1}",
                        running_loss / running_data_num,
                    )

            self.server.receive(use_gradients=self.use_gradients)
            if self.use_gradients:
                self.server.updata_from_gradients(weight=self.clients_weight)
            else:
                self.server.update_from_parameters(weight=self.clients_weight)

            self.custom_action(self)


class MPIFedProxAPI(MPIFedAVGAPI):
    def __init__(self, *args, mu=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu

    def local_train(self):
        self.party.prev_parameters = []
        for param in self.party.model.parameters():
            self.party.prev_parameters.append(copy.deepcopy(param))

        for _ in range(self.local_epoch):
            running_loss = 0
            for (data, target) in self.local_dataloader:
                self.local_optimizer.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.party.model(data)

                loss = self.criterion(output, target)
                loss.backward()

                for local_param, global_param in zip(
                    self.party.model.parameters(), self.party.prev_parameters
                ):
                    loss += (
                        self.mu / 2 * torch.norm(local_param.data - global_param.data)
                    )
                    local_param.grad.data += self.mu * (
                        local_param.data - global_param.data
                    )

                self.local_optimizer.step()
                running_loss += loss.item()
