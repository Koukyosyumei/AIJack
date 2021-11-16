import copy

import torch

from ..base_attack import BaseAttacker


class GAN_Attack(BaseAttacker):
    def __init__(
        self,
        client,
        target_label,
        generator,
        generator_optimizer,
        generator_criterion,
        nz=100,
        device="cpu",
    ):
        """Implementation of model inversion attack
           reference https://arxiv.org/abs/1702.07464

        Args:
            model (torch.nn.Module):
            target_label(int): index of target class
            generator (torch.nn.Module): Generator
            generator_optimizer (torch.optim.Optimizer): optimizer for the generator
            generator_criterion (function): loss function for the generator
            nz (int): dimension of latent space of the generator
            user_id (int): user id
            device (string): device type (cpu or cuda)
        """
        super().__init__(target_model=None)
        self.client = client
        self.target_label = target_label
        self.generator = generator
        self.generator_optimizer = generator_optimizer
        self.generator_criterion = generator_criterion
        self.nz = nz
        self.device = device

        self.discriminator = copy.deepcopy(self.client.model)
        self.discriminator.to(self.device)

        self.noise = torch.randn(1, self.nz, 1, 1, device=self.device)

    def update_generator(self, batch_size=10, epoch=1, log_interval=5):
        """Updata the Generator

        Args:
            batch_size (int): batch size
            epoch (int): epoch
            log_interval (int): interval of logging
        """

        for i in range(epoch):
            running_error = 0
            self.generator.zero_grad()

            noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
            fake = self.generator(noise)
            output = self.discriminator(fake)

            label = torch.full(
                (batch_size,), self.target_label, dtype=torch.int64, device=self.device
            )
            loss_generator = self.generator_criterion(output, label)
            loss_generator.backward()

            self.generator_optimizer.step()

            running_error += loss_generator.item()

            if i % log_interval == 0:
                print(
                    f"updating generator - epoch {i}: generator loss is {running_error/batch_size}"
                )

    def update_discriminator(self):
        """Update the discriminator"""
        self.discriminator.load_state_dict(self.client.model.state_dict())

    def attack(self, n):
        """Generate fake images

        Args:
            n (int): the number of fake images created by the Generator

        Returns:
            fake: generated fake images
        """
        noise = torch.randn(n, self.nz, 1, 1, device=self.device)
        with torch.no_grad():
            fake = self.generator(noise)
        return fake
