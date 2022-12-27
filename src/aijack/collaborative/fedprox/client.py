import torch

from ..fedavg import FedAVGClient


class FedProxClient(FedAVGClient):
    def local_train(
        self,
        server_parameters,
        local_epoch,
        criterion,
        trainloader,
        optimizer,
        communication_id=0,
    ):
        for i in range(local_epoch):
            running_loss = 0.0
            running_data_num = 0
            for _, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                inputs.requires_grad = True
                labels = labels.to(self.device)

                optimizer.zero_grad()
                self.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                for local_param, global_param in zip(
                    self.parameters(), server_parameters
                ):
                    loss += (
                        self.mu / 2 * torch.norm(local_param.data - global_param.data)
                    )
                    local_param.grad.data += self.mu * (
                        local_param.data - global_param.data
                    )

                optimizer.step()

                running_loss += loss.item()
                running_data_num += inputs.shape[0]

            print(
                f"communication {communication_id}, epoch {i}: client-{self.user_id+1}",
                running_loss / running_data_num,
            )
