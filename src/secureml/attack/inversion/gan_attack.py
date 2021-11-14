import copy

import torch

from ...collaborative import Client


class GAN_Attack_Client(Client):
    def __init__(
        self,
        model,
        target_label,
        generator,
        generator_optimizer,
        generator_criterion,
        nz=100,
        user_id=0,
        device="cpu",
    ):
        super().__init__(model, user_id=user_id)
        self.target_label = target_label
        self.generator = generator
        self.generator_optimizer = generator_optimizer
        self.generator_criterion = generator_criterion
        self.nz = nz
        self.device = device

        self.discriminator = copy.deepcopy(model)

    def update_generator(self, dataloader, log_interval=5):
        for i, data in enumerate(dataloader, 0):
            self.generator.zero_grad()
            real_cpu = data[0].to(self.device)
            b_size = real_cpu.size(0)
            label = torch.full(
                (b_size,), self.target_label, dtype=torch.float, device=self.device
            )
            noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
            fake = self.generator(noise).view(-1)
            output = self.discriminator(fake).view(-1)
            loss_generator = self.generator_criterion(output, label)
            loss_generator.backward()
            self.generator_optimizer.step()

            if i % log_interval == 0:
                print(f"{i} - {loss_generator.item()}")
