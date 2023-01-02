import copy

import torch

from ...manager import BaseManager


def attach_ganattack_to_client(
    cls,
    target_label,
    generator,
    generator_optimizer,
    generator_criterion,
    nz=100,
    device="cpu",
    gan_batch_size=1,
    gan_epoch=1,
    gan_log_interval=0,
    ignore_first_download=False,
):
    """Wraps the given class in GANAttackClientWrapper.

    Args:
        target_label(int): index of target class
        generator (torch.nn.Module): Generator
        generator_optimizer (torch.optim.Optimizer): optimizer for the generator
        generator_criterion (function): loss function for the generator
        nz (int): dimension of latent space of the generator. Defaults to 100.
        device (str, optional): _description_. Defaults to "cpu".
        gan_batch_size (int, optional): batch size for training GAN. Defaults to 1.
        gan_epoch (int, optional): epoch for training GAN. Defaults to 1.
        gan_log_interval (int, optional): log interval. Defaults to 0.
        ignore_first_download (bool, optional): Defaults to False.

    Returns:
        cls: a class wrapped in GANAttackClientWrapper
    """

    class GANAttackClientWrapper(cls):
        """Implementation of GAN based model inversion attack (https://arxiv.org/abs/1702.07464)"""

        def __init__(self, *args, **kwargs):
            super(GANAttackClientWrapper, self).__init__(*args, **kwargs)

            self.target_label = target_label
            self.generator = generator
            self.generator_optimizer = generator_optimizer
            self.generator_criterion = generator_criterion
            self.nz = nz
            self.device = device

            self.discriminator = copy.deepcopy(self.model)
            self.discriminator.to(self.device)

            self.noise = torch.randn(1, self.nz, 1, 1, device=self.device)
            self.is_params_initialized = False

        def update_generator(self, batch_size=10, epoch=1, log_interval=5):
            """Updates the Generator

            Args:
                batch_size (int): batch size
                epoch (int): epoch
                log_interval (int): interval of logging
            """

            for i in range(1, epoch + 1):
                running_error = 0
                self.generator.zero_grad()

                noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
                fake = self.generator(noise)
                output = self.discriminator(fake)

                label = torch.full(
                    (batch_size,),
                    self.target_label,
                    dtype=torch.int64,
                    device=self.device,
                )
                loss_generator = self.generator_criterion(output, label)
                loss_generator.backward()

                self.generator_optimizer.step()

                running_error += loss_generator.item()

                if log_interval != 0 and i % log_interval == 0:
                    print(
                        f"updating generator - epoch {i}: generator loss is {running_error/batch_size}"
                    )

        def update_discriminator(self):
            """Updates the discriminator"""
            self.discriminator.load_state_dict(self.model.state_dict())

        def download(self, model_parameters):
            """Downloads the new model"""
            super().download(model_parameters)
            if ignore_first_download and not self.is_params_initialized:
                self.is_params_initialized = True
                return
            self.update_discriminator()
            self.update_generator(
                batch_size=gan_batch_size,
                epoch=gan_epoch,
                log_interval=gan_log_interval,
            )

        def attack(self, n):
            """Generates fake images

            Args:
                n (int): the number of fake images created by the Generator

            Returns:
                fake: generated fake images
            """
            noise = torch.randn(n, self.nz, 1, 1, device=self.device)
            with torch.no_grad():
                fake = self.generator(noise)
            return fake

    return GANAttackClientWrapper


class GANAttackClientManager(BaseManager):
    """Manager class for GAN based model inversion attack (https://arxiv.org/abs/1702.07464)"""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def attach(self, cls):
        """Wraps the given class in GANAttackClientWrapper.

        Returns:
            cls: a class wrapped in GANAttackClientWrapper
        """
        return attach_ganattack_to_client(cls, *self.args, **self.kwargs)
