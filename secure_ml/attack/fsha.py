import torch


class FSHA:
    def __init__(self,
                 client_dataloader,
                 attacker_dataloader,
                 f, tilde_f, D, decoder,
                 optimizers,
                 wgan=True,
                 gradient_penalty=100,
                 distance_data_loss=torch.nn.MSELoss(),
                 distance_data=torch.nn.MSELoss(),
                 ):
        """
        Args
            client_dataloader: dataloader of client's private dataset
            attacker_dataloader: dataloader of attacker's public dataset
            f: client's network
            tilde_f: pilot network that dynamically defines the target
                     feature-space Z˜ for the client’s network
            D: discriminator that indirectly guides 𝑓 to learn
               a mapping between the private data and the feature-space
               defined from tilde_f
            decoder:  approximation of the inverse function of tilde_f
            optimizers: list of optimizers
                        optimizers[0] is for f
                        optimizers[1] is for tilde_f and decoder
                        optimizers[2] is for D
            wgan:
            gradient_penalty:
            distance_data_loss: loss function to update tilde_f
            distance_data: function to culculate the distance between
                           original data and reconstruction data

        Attributes
            client_dataloader: dataloader of client's private dataset
            attacker_dataloader: dataloader of attacker's public dataset
            f: client's network
            tilde_f: pilot network that dynamically defines the target
                     feature-space Z˜ for the client’s network
            D: discriminator that indirectly guides 𝑓 to learn
               a mapping between the private data and the feature-space
               defined from tilde_f
            decoder:  approximation of the inverse function of tilde_f
            optimizers0: optimizer for f
            optimizers1: optimizer for tilde_f and decoder
            optimizers2: optimizer for D
            wgan:
            gradient_penalty:
            distance_data_loss: loss function to update tilde_f
            distance_data: function to culculate the distance between
                           original data and reconstruction data
        """

        self.client_dataloader = client_dataloader
        self.attacker_dataloader = attacker_dataloader

        self.f = f
        self.tilde_f = tilde_f
        self.decoder = decoder
        self.D = D

        self.wgan = wgan
        self.w = gradient_penalty

        # TODO: check the shape of output
        # assert the shape of f's input == the shape of data
        # assert the shape of f's input == the shape of tilde_f's input
        # assert the shape of f's output == the shape of tilde_f's output
        # assert the shape of D's input == tilde_f's output
        # assert the shape of decoder's input == the shape of tilde_f's output

        assert len(optimizers) == 3, "length of optimizers must be three."
        self.optimizer0 = optimizers[0](self.f.parameters(), lr=0.00001)
        self.optimizer1 = optimizers[1](list(self.tilde_f.parameters()) +
                                        list(self.decoder.parameters()),
                                        lr=0.00001)
        self.optimizer2 = optimizers[2](self.D.parameters(), lr=0.00001)

        self.distance_data_loss = distance_data_loss
        self.distance_data = distance_data

    @staticmethod
    def addNoise(x, alpha):
        pass

    def train(self, epochs=10, verbose=2, save_log=False):
        """
        Args:
            epochs (int): num of iteration
            verbose (int): log interval
            save_log (bool):

        Returns:
            log
        """

        log = {"f_loss": [],
               "tilde_f_loss": [],
               "D_loss": [],
               "loss_c_verification": []}

        len_dataloader = len(self.client_dataloader)
        for epoch in range(epochs):

            epoch_f_loss = 0
            epoch_tilde_f_loss = 0
            epoch_D_loss = 0
            epoch_loss_c_verification = 0

            for (x_private, label_private), (x_public, label_public) in\
                    zip(self.client_dataloader, self.attacker_dataloader):
                f_loss, tilde_f_loss,\
                    D_loss, loss_c_verification\
                    = self.train_step(x_private, x_public,
                                      label_private, label_public)

                epoch_f_loss += f_loss / len_dataloader
                epoch_tilde_f_loss += tilde_f_loss / len_dataloader
                epoch_D_loss += D_loss / len_dataloader
                epoch_loss_c_verification +=\
                    loss_c_verification / len_dataloader

            if save_log:
                log["f_loss"].append(epoch_f_loss)
                log["tilde_f_loss"].append(epoch_tilde_f_loss)
                log["D_loss"].append(epoch_D_loss)
                log["loss_c_verification"].append(epoch_loss_c_verification)

            if epoch % verbose == 0:
                print(f"f_loss:{epoch_f_loss} " +
                      f"tilde_f_loss:{epoch_tilde_f_loss} " +
                      f"D_loss:{epoch_D_loss} " +
                      f"loss_c:{epoch_loss_c_verification}")

        return log

    def train_step(self, x_private, x_public, label_private, label_public):
        """
        Args:
            x_private (torch.Tensor):
            x_public (torch.Tensor):
            label_private (torch.Tensor):
            label_public (torch.Tensor):

        Return:
            f_loss:
            tilde_f_loss:
            D_loss:
            loss_c_verification:
        """

        # initialize optimizers
        self.optimizer0.zero_grad()
        # Virtually, ON THE CLIENT SIDE:
        # clients' smashed data
        z_private = self.f(x_private)
        ####################################

        # SERVER-SIDE:
        # adversarial loss (f's output must similar be to \tilde{f}'s output):
        adv_private_logits = self.D(z_private)

        if self.wgan:
            f_loss = torch.mean(adv_private_logits)
        else:
            f_loss = torch.nn.BCEWithLogitsLoss()(
                torch.ones_like(adv_private_logits), adv_private_logits)
        f_loss.backward()
        self.optimizer0.step()

        # update tilde_f and decoder
        self.optimizer1.zero_grad()
        z_public = self.tilde_f(x_public)
        rec_x_public = self.decoder(z_public)
        # invertibility loss
        tilde_f_loss = self.distance_data_loss(x_public, rec_x_public)
        tilde_f_loss.backward()
        self.optimizer1.step()

        # updata D
        self.optimizer2.zero_grad()

        with torch.no_grad():
            z_private_ = z_private + 0
            z_public_ = z_public + 0
        adv_private_logits = self.D(z_private_)
        adv_public_logits = self.D(z_public_)

        if self.wgan:
            loss_discr_true = torch.mean(adv_public_logits)
            loss_discr_fake = -torch.mean(adv_private_logits)
            vanila_D_loss = loss_discr_true + loss_discr_fake
        else:
            loss_discr_true = torch.nn.BCEWithLogitsLoss()(
                torch.ones_like(adv_public_logits), adv_public_logits)
            loss_discr_fake = torch.nn.BCEWithLogitsLoss()(
                torch.zeros_like(adv_private_logits), adv_private_logits)
            # discriminator's loss
            vanila_D_loss = (loss_discr_true + loss_discr_fake) / 2

        D_loss = vanila_D_loss + self.w*self.gradient_penalty(z_private,
                                                              z_public)
        D_loss.backward()
        self.optimizer2.step()

        # evaluation
        # map to data space (for evaluation and style loss)
        rec_x_private = self.decoder(z_private)
        loss_c_verification = self.distance_data(x_private, rec_x_private)

        return f_loss, tilde_f_loss, vanila_D_loss, loss_c_verification

    def gradient_penalty(self, x, x_gen):
        """
        Args:
            x
            x_gen

        Returns:
            d_regularizer
        """
        epsilon = torch.randn(x.shape[0], 1, 1, 1)
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        d_hat = self.D(x_hat)
        gradients = torch.autograd.grad(outputs=d_hat,
                                        grad_outputs=torch.ones(d_hat.size()),
                                        inputs=x_hat,
                                        # create_graph=True,
                                        retain_graph=True
                                        )[0]
        ddx = torch.sqrt(torch.mean(gradients.pow(2), dim=(1, 2)))
        d_regularizer = torch.mean((ddx-1).pow(2))

        return d_regularizer

    def attack(self, x_private):
        """attack splitNN
        Args:
            x_private (torch.Tensor): client's dataset

        Returns:
            tilde_x_private (torch.Tensor): self.decoder(self.f(x_private))
            control (torch.Tensor): self.decoder(self.tilde_f(x_private))
        """
        with torch.no_grad():
            # smashed data sent from the client:
            z_private = self.f(x_private)
            # recover private data from smashed data
            tilde_x_private = self.decoder(z_private)

            z_private_control = self.tilde_f(x_private)
            control = self.decoder(z_private_control)

        return tilde_x_private, control

    def score(self, x_private, label_private):
        pass

    def scoreAttack(self, dataset):
        pass
