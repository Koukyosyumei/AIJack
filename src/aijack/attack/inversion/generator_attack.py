import torch

from ..base_attack import BaseAttacker


class Generator_Attack(BaseAttacker):
    def __init__(self, target_model, attacker_model, attacker_optimizer):
        super().__init__(target_model=target_model)
        self.attacker_model = attacker_model
        self.attacker_optimizer = attacker_optimizer

    def fit(self, dataloader_for_attacker, epoch):

        for i in range(epoch):
            for data, _ in dataloader_for_attacker:
                self.attacker_optimizer.zero_grad()

                target_outputs = self.target_model(data)

                attack_outputs = self.attacker_model(target_outputs)

                loss = ((data - attack_outputs) ** 2).mean()

                loss.backward()
                self.attacker_optimizer.step()

            print(f"epoch {i}: reconstruction_loss {loss.item()}")

    def attack(self, dataloader_target):
        attack_results = []

        for data, _ in dataloader_target:
            target_outputs = self.target_model(data)
            recreated_data = self.attacker_model(target_outputs)
            attack_results.append(recreated_data)

        return torch.cat(attack_results)
