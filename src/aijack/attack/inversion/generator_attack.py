from ..base_attack import BaseAttacker


class Generator_Attack(BaseAttacker):
    def __init__(
        self, target_model, attacker_model, attacker_optimizer, log_interval=1
    ):
        super().__init__(target_model=target_model)
        self.attacker_model = attacker_model
        self.attacker_optimizer = attacker_optimizer
        self.log_interval = log_interval

    def fit(self, dataloader, epoch, x_pos=0):
        for i in range(epoch):
            for data in dataloader:
                x = data[x_pos]
                self.attacker_optimizer.zero_grad()
                target_outputs = self.target_model(x)
                attack_outputs = self.attacker_model(target_outputs)
                loss = ((x - attack_outputs) ** 2).mean()
                loss.backward()
                self.attacker_optimizer.step()

            if self.log_interval != 0 and i % self.log_interval == 0:
                print(f"epoch {i}: reconstruction_loss {loss.item()}")

    def attack(self, data_tensor):
        return self.attacker_model(data_tensor)
