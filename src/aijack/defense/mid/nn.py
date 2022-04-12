import torch
from torch import nn

from .loss import mib_loss


class VIB(nn.Module):
    def __init__(self, encoder, decoder, dim_z=256, num_samples=10, beta=1e-3):
        super(VIB, self).__init__()
        self.dim_z = dim_z
        self.num_samples = num_samples
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

    def get_params_of_p_z_given_x(self, x):
        encoder_output = self.encoder(x)
        if encoder_output.shape[1] != self.dim_z * 2:
            raise ValueError("the output dimension of encoder must be 2 * dim_z")
        mu = encoder_output[:, : self.dim_z]
        sigma = torch.nn.functional.softplus(encoder_output[:, self.dim_z :])
        return mu, sigma

    def sampling_from_encoder(self, mu, sigma, batch_size):
        return mu + sigma * torch.normal(
            torch.zeros(self.num_samples, batch_size, self.dim_z),
            torch.ones(self.num_samples, batch_size, self.dim_z),
        )

    def forward(self, x):
        batch_size = x.size()[0]

        # encoder
        p_z_given_x_mu, p_z_given_x_sigma = self.get_params_of_p_z_given_x(x)
        sampled_encoded_features = self.sampling_from_encoder(
            p_z_given_x_mu, p_z_given_x_sigma, batch_size
        )

        # decoder
        sampled_decoded_outputs = self.decoder(sampled_encoded_features)
        outputs = torch.mean(sampled_decoded_outputs, dim=0)

        if self.training:
            return outputs, {
                "sampled_decoded_outputs": sampled_decoded_outputs.permute(1, 2, 0),
                "sampled_encoded_features": sampled_encoded_features,
                "p_z_given_x_mu": p_z_given_x_mu,
                "p_z_given_x_sigma": p_z_given_x_sigma,
            }
        else:
            return outputs

    def loss(self, y, result_dict):
        sampled_y_pred = result_dict["sampled_decoded_outputs"]
        p_z_given_x_mu = result_dict["p_z_given_x_mu"]
        p_z_given_x_sigma = result_dict["p_z_given_x_sigma"]

        approximated_z_mean = torch.zeros_like(p_z_given_x_mu)
        approximated_z_sigma = torch.ones_like(p_z_given_x_sigma)

        loss, _, _ = mib_loss(
            y,
            sampled_y_pred,
            p_z_given_x_mu,
            p_z_given_x_sigma,
            approximated_z_mean,
            approximated_z_sigma,
            beta=self.beta,
        )

        return loss
