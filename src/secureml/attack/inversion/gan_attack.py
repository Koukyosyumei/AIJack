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
        self.discriminator.to(self.device)

        self.noise = torch.randn(1, self.nz, 1, 1, device=self.device)

    def update_generator(self, dataloader, epoch=1, log_interval=5):
        for i in range(epoch):
            running_error = 0
            data_size = 0

            for _, data in enumerate(dataloader, 0):
                self.generator.zero_grad()
                real = data[0].to(self.device)
                b_size = real.size(0)
                data_size += b_size

                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                fake = self.generator(noise)
                output = self.discriminator(fake)

                label = torch.full(
                    (b_size,), self.target_label, dtype=torch.int64, device=self.device
                )
                loss_generator = self.generator_criterion(output, label)
                loss_generator.backward()

                self.generator_optimizer.step()

                running_error += loss_generator.item()

            if i % log_interval == 0:
                print(f"updating generator - epoch {i}: generator loss is {running_error/data_size}")

    def download(self, model_parameters):
        self.model.load_state_dict(model_parameters)
        self.discriminator.load_state_dict(model_parameters)

    def attack(self, n):
        noise = torch.randn(n, self.nz, 1, 1, device=self.device)
        with torch.no_grad():
            fake = self.generator(noise)
        return fake
