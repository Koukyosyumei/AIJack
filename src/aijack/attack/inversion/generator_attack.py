from ..base_attack import BaseAttacker


class Generator_Attack(BaseAttacker):
    def __init__(
        self,
        target_model,
        attacker_model,
        attacker_optimizer,
        log_interval=1,
        early_stopping=5,
    ):
        super().__init__(target_model=target_model)
        self.attacker_model = attacker_model
        self.attacker_optimizer = attacker_optimizer
        self.log_interval = log_interval
        self.early_stopping = early_stopping

    def fit(self, dataloader, epoch, x_pos=0):
        best_loss = float("inf")
        best_epoch = 0
        for i in range(epoch):
            running_loss = 0
            for data in dataloader:
                x = data[x_pos]
                self.attacker_optimizer.zero_grad()
                target_outputs = self.target_model(x)
                attack_outputs = self.attacker_model(target_outputs)
                loss = ((x - attack_outputs) ** 2).mean()
                loss.backward()
                self.attacker_optimizer.step()
                running_loss += loss.item() / len(dataloader)

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
