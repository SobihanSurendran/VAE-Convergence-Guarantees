import torch
from typing import Tuple, Dict, Any, Optional
import os
import torch.nn as nn
import torch.nn.functional as F

from outils import get_encoder_decoder


class VAE(nn.Module):
    def __init__(self, model_params: Dict[str, Any]) -> None:
        """
        Initializes the Variational Autoencoder (VAE) model.

        Args:
            model_params (Dict[str, Any]): A dictionary containing model hyperparameters.
                - 'dataset': The dataset used for training the VAE.
                - 'latent_size': Dimensionality of the latent space.
                - 'beta': A weighting factor for the KL divergence term in the ELBO.
                - 'soft_clipping_activation': Indicates whether soft clipping activation is enabled in the encoder/decoder.
                - 's': A hyperparameter that controls the sharpness of the transition in the soft clipping activation.
        """
        super(VAE, self).__init__()
        self.dataset = model_params['dataset']
        self.latent_size = model_params['latent_size']
        self.beta = model_params['beta']
        self.soft_cipping_activation = model_params['soft_cipping_activation']
        self.s = model_params['s']
        self.encoder, self.fc_mu, self.fc_logvar, self.decoder = get_encoder_decoder(self.dataset, self.latent_size, self.soft_cipping_activation, self.s)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Defines the full forward pass through the VAE.

        Args:
        x (torch.Tensor): Input tensor (batch of data) to be encoded and decoded.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - x_hat (torch.Tensor): The reconstructed output generated by the decoder.
            - mu (torch.Tensor): The mean of the variational distribution.
            - logvar (torch.Tensor): The log variance of the variational distribution.
            - z (torch.Tensor): Latent samples drawn from the variational distribution using the reparameterization trick.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the parameters of the variational distribution: mean and log variance.

        Args:
            x (torch.Tensor): Input tensor (batch of data) to be encoded.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - mu (torch.Tensor): The mean of the variational distribution.
                - logvar (torch.Tensor): The log variance of the variational distribution.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Applies the reparameterization trick to sample from the variational distribution N(mu, sigma^2).

        Args:
            mu (torch.Tensor): The mean of the variational distribution.
            logvar (torch.Tensor): The log variance of the variational distribution.

        Returns:
            torch.Tensor: Latent samples drawn from the variational distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent samples z back into the original image space.

        Args:
            z (torch.Tensor): Latent samples drawn from the variational distribution.

        Returns:
            torch.Tensor: Reconstructed output tensor from the decoder.
        """
        x_hat = self.decoder(z)
        return x_hat

    def loss(self, x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the VAE loss, including reconstruction loss and KL divergence.

        Args:
            x_hat (torch.Tensor): The reconstructed output generated by the decoder.
            x (torch.Tensor): The original input tensor, serving as the target for reconstruction.
            mu (torch.Tensor): The mean of the variational distribution.
            logvar (torch.Tensor): The log variance of the variational distribution.
            z (torch.Tensor): Latent samples drawn from the variational distribution.

        Returns:
            torch.Tensor: The total loss value, combining reconstruction loss and KL divergence.
        """
        reconstruction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + self.beta*kl_divergence

