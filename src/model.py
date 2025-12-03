from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int
    image_size: Tuple[int, int]
    embed_dim: int = 64
    text_hidden_dim: int = 128
    latent_dim: int = 128


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text)
        _, hidden = self.gru(embedded)
        return hidden.squeeze(0)


class ConvEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, text_dim: int, latent_dim: int, flattened_size: int):
        super().__init__()
        self.conv = backbone
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(flattened_size + text_dim, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size + text_dim, latent_dim)

    def forward(self, image: torch.Tensor, text_features: torch.Tensor):
        x = self.conv(image)
        x = self.flatten(x)
        x = torch.cat([x, text_features], dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class ConvDecoder(nn.Module):
    def __init__(self, text_dim: int, latent_dim: int, conv_shape: Tuple[int, int, int]):
        super().__init__()
        c, h, w = conv_shape
        self.conv_shape = conv_shape
        self.fc = nn.Linear(latent_dim + text_dim, c * h * w)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, text_features: torch.Tensor):
        x = torch.cat([z, text_features], dim=1)
        x = self.fc(x)
        b = x.size(0)
        c, h, w = self.conv_shape
        x = x.view(b, c, h, w)
        x = self.decoder(x)
        return x


class ConditionalVAE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.text_encoder = TextEncoder(config.vocab_size, config.embed_dim, config.text_hidden_dim)

        self.backbone = self._build_backbone()
        conv_shape = self._infer_conv_shape(config.image_size)
        flattened = conv_shape[0] * conv_shape[1] * conv_shape[2]

        self.encoder = ConvEncoder(self.backbone, config.text_hidden_dim, config.latent_dim, flattened)
        self.decoder = ConvDecoder(config.text_hidden_dim, config.latent_dim, conv_shape)
        self.latent_dim = config.latent_dim

    def _build_backbone(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _infer_conv_shape(self, image_size: Tuple[int, int]) -> Tuple[int, int, int]:
        dummy = torch.zeros(1, 1, *image_size)
        with torch.no_grad():
            out = self.backbone(dummy)
        return out.shape[1], out.shape[2], out.shape[3]

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        text_features = self.text_encoder(text)
        mu, logvar = self.encoder(image, text_features)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, text_features)
        return recon, mu, logvar

    def generate(self, text: torch.Tensor, device: torch.device, latent: torch.Tensor | None = None):
        text_features = self.text_encoder(text)
        batch_size = text.size(0)
        if latent is None:
            latent = torch.randn(batch_size, self.encoder.fc_mu.out_features, device=device)
        with torch.no_grad():
            images = self.decoder(latent, text_features)
        return images


class VAELoss(nn.Module):
    def __init__(self, recon_weight: float = 1.0, kl_weight: float = 0.0005):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.criterion = nn.L1Loss()

    def forward(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        recon_loss = self.criterion(recon_x, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return self.recon_weight * recon_loss + self.kl_weight * kl_loss, recon_loss, kl_loss
