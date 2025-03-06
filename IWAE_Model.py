import torch
from typing import Tuple, Dict, Any, Optional
import os
import torch.nn as nn
import torch.nn.functional as F

from outils import get_encoder_decoder

class IWAE(nn.Module):
    def __init__(self, model_params: Dict[str, Any]) -> None:
        super(IWAE, self).__init__()
        self.dataset = model_params['dataset']
        self.latent_size = model_params['latent_size']
        self.K = model_params['K']
        self.soft_cipping_activation = model_params['soft_cipping_activation']
        self.s = model_params['s']
        self.encoder, self.fc_mu, self.fc_logvar, self.decoder = get_encoder_decoder(self.dataset, self.latent_size, self.soft_cipping_activation, self.s)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(self.K, *mu.size(), device=mu.device)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[1]
        z = z.view(-1, self.latent_size)
        x_hat = self.decoder(z)
        x_hat = x_hat.view(self.K, batch_size, 3, 32, 32)
        return x_hat

    def loss(self, x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        log_q_z_x = torch.distributions.Normal(loc=mu, scale=std).log_prob(z).sum(-1)
        log_p_z = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std)).log_prob(z).sum(-1)
        xx = x.unsqueeze(0).repeat(self.K, 1, 1, 1, 1)
        likelihood = -F.binary_cross_entropy(x_hat, xx, reduction='none').sum(dim=[2, 3, 4])

        log_weight = log_p_z + likelihood - log_q_z_x
        log_weight2 = log_weight - torch.max(log_weight, 0)[0]
        weight = torch.exp(log_weight2)
        weight = weight / torch.sum(weight, 0)
        weight = weight.detach()

        loss = -torch.sum(torch.sum(weight * log_weight, 0)) + torch.log(torch.Tensor([self.K]).to(mu.device))
        return loss

