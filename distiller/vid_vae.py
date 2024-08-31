import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VIDVAELoss(nn.Module):
    def __init__(self, num_input_channels, num_latent_dim, num_target_channels, init_pred_var=5.0, eps=1e-5):
        super(VIDVAELoss, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 15 * 12, 256),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(256, num_latent_dim)
        self.fc_logvar = nn.Linear(256, num_latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(num_latent_dim, 128 * 16 * 12),
            nn.ReLU(),
            nn.Unflatten(1, (128, 16, 12)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_target_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )


        self.regressor = nn.Sequential(
            nn.Conv2d(num_input_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, num_target_channels, kernel_size=1),
        )

        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var - eps) - 1.0) * torch.ones(num_target_channels)
        )
        self.eps = eps

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input, target):
        s_H, t_H = input.shape[2], target.shape[2]
        if s_H > t_H:
            input = F.adaptive_avg_pool2d(input, (t_H, t_H))
        elif s_H < t_H:
            target = F.adaptive_avg_pool2d(target, (s_H, s_H))

        encoded = self.encoder(input)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)

        reconstructed = self.decoder(z)
        reconstructed = F.interpolate(reconstructed, size=(target.shape[2], target.shape[3]), mode='bilinear', align_corners=False)

        target = target.clone()
        reconstructed = reconstructed.clone()

        recon_loss = F.mse_loss(reconstructed, target, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0 + torch.exp(self.log_scale)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)

        neg_log_prob = 0.5 * (
            (pred_mean - target) ** 2 / pred_var + torch.log(pred_var)
        )
        vid_loss = torch.mean(neg_log_prob)

        total_loss = vid_loss + recon_loss + kl_loss
        return total_loss