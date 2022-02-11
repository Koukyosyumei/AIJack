import torch

from ..base_attack import BaseAttacker


class Generator_Attack(BaseAttacker):
    def __init__(
        self, target_model, attacker_model, attacker_optimizer, log_interval=1
    ):
        super().__init__(target_model=target_model)
        self.attacker_model = attacker_model
        self.attacker_optimizer = attacker_optimizer
        self.log_interval = log_interval

    def fit(self, dataloader, epoch):

        for i in range(epoch):
            for data in dataloader:
                self.attacker_optimizer.zero_grad()
                target_outputs = self.target_model(data)
                attack_outputs = self.attacker_model(target_outputs)
                loss = ((data - attack_outputs) ** 2).mean()
                loss.backward()
                self.attacker_optimizer.step()

            if self.log_interval != 0 and i % self.log_interval == 0:
                print(f"epoch {i}: reconstruction_loss {loss.item()}")

    def attack(self, dataloader):
        attack_results = []

        for data in dataloader:
            recreated_data = self.attacker_model(data)
            attack_results.append(recreated_data)

        return torch.cat(attack_results)
