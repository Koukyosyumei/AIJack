from typing import List

from ..base_attack import BaseAttacker


class Generator_Attack(BaseAttacker):
    def __init__(
        self,
        target_model,
        attacker_model,
        attacker_optimizer,
        log_interval=1,
        early_stopping=5,
        device="cpu",
    ):
        if type(target_model) == List:
            super().__init__(target_model=target_model[0])
        else:
            super().__init__(target_model=target_model)
        self.attacker_model = attacker_model
        self.attacker_optimizer = attacker_optimizer
        self.log_interval = log_interval
        self.early_stopping = early_stopping
        self.device = device

        self.target_model_list = (
            target_model if type(target_model) == list else [target_model]
        )

    def culc_loss(self, dataloader, x_pos=0, y_pos=1, arbitrary_y=False):
        running_loss = 0
        for data in dataloader:
            x = data[x_pos]
            x = x.to(self.device)

            loss = 0
            for target_model in self.target_model_list:
                target_outputs = data[y_pos] if arbitrary_y else target_model(x)
                target_outputs = target_outputs.to(self.device)

                attack_outputs = self.attacker_model(target_outputs)
                loss = loss + ((x - attack_outputs) ** 2).mean()

            running_loss = running_loss + loss / len(dataloader)

        return running_loss

    def fit(self, dataloader, epoch, x_pos=0, y_pos=1, arbitrary_y=False):
        best_loss = float("inf")
        best_epoch = 0
        for i in range(epoch):
            running_loss = 0
            for data in dataloader:
                x = data[x_pos]
                x = x.to(self.device)

                loss = 0
                self.attacker_optimizer.zero_grad()
                for target_model in self.target_model_list:
                    target_outputs = data[y_pos] if arbitrary_y else target_model(x)
                    target_outputs = target_outputs.to(self.device)

                    attack_outputs = self.attacker_model(target_outputs)
                    loss = loss + ((x - attack_outputs) ** 2).mean() / len(
                        self.target_model_list
                    )
                loss.backward()
                self.attacker_optimizer.step()

                running_loss = running_loss + loss.item() / len(dataloader)

            if self.log_interval != 0 and i % self.log_interval == 0:
                print(f"epoch {i}: reconstruction_loss {running_loss}")

            if running_loss < best_loss:
                best_loss = running_loss
                best_epoch = i
            else:
                if i - best_epoch > self.early_stopping:
                    break

    def attack(self, data_tensor):
        return self.attacker_model(data_tensor)
