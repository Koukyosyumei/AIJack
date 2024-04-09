import copy

import torch
import torch.nn.functional as F

from ..fedavg import FedAVGClient


class MOONClient(FedAVGClient):
    """Client of MOON for single process simulation
    (Li, Qinbin, Bingsheng He, and Dawn Song. "Model-contrastive
    federated learning." Proceedings of the IEEE/CVF conference
    on computer vision and pattern recognition. 2021.)

    Args:
        model (torch.nn.Module): local model
        mu (float): weight of model-contrastive loss
        tau (float): tempreature within model-contrastive loss
    """

    def __init__(
        self,
        model,
        mu=0.1,
        tau=1.0,
        **kwargs,
    ):
        super(MOONClient, self).__init__(model, **kwargs)
        self.mu = mu
        self.tau = tau
        self.global_model = copy.deepcopy(model)
        self.prev_model = copy.deepcopy(model)

    def local_train(
        self,
        local_epoch,
        criterion,
        trainloader,
        optimizer,
        communication_id=0,
    ):
        if communication_id != 0:
            for param, glob_param in zip(
                self.global_model.parameters(), self.model.parameters()
            ):
                if param is not None:
                    param = glob_param
            for param, prev_param in zip(
                self.prev_model.parameters(), self.prev_parameters
            ):
                if param is not None:
                    param = prev_param

        loss_log = []

        for _ in range(local_epoch):
            running_loss = 0.0
            running_data_num = 0
            for _, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                self.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)

                if communication_id != 0:
                    glob_outputs = self.global_model(inputs)
                    prev_outputs = self.prev_model(inputs)

                    exp_sim_cg = torch.exp(
                        F.cosine_similarity(outputs, glob_outputs) / self.tau
                    )
                    exp_sim_cp = torch.exp(
                        F.cosine_similarity(outputs, prev_outputs) / self.tau
                    )
                    loss_con = -1 * torch.log(exp_sim_cg / (exp_sim_cg + exp_sim_cp))
                    loss = loss + self.mu * loss_con

                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                running_data_num += inputs.shape[0]

            loss_log.append(running_loss / running_data_num)

        return loss_log
