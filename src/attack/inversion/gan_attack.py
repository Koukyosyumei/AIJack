import copy

from ...collaborative import Client


class GAN_Attack_Client(Client):
    def __init__(self, model, discriminator, user_id=0):
        super().__init__(model, user_id=user_id)
        self.discriminator = discriminator
        self.generator = copy.deepcopy(model)

    def update_gan(self):
        pass
