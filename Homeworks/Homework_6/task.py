import torch
from torch import nn
from torch.nn import functional as F

# Task 1

class Block(nn.Module):
    def __init__(self, start_channels, flag_up=False):
        super().__init__()
        if not flag_up:
            self.conv = nn.Conv2d(in_channels=start_channels, out_channels=start_channels * 2, kernel_size=3, stride=2,
                                  padding=1)
            self.batch_norm = nn.BatchNorm2d(start_channels * 2)

        else:
            self.conv = nn.ConvTranspose2d(in_channels=start_channels, out_channels=start_channels // 2, kernel_size=4,
                                           stride=2, padding=1)
            self.batch_norm = nn.BatchNorm2d(start_channels // 2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, start_channels=16, downsamplings=5):
        super().__init__()
        self.first_conv = nn.Conv2d(in_channels=3, out_channels=start_channels, kernel_size=1, stride=1, padding=0)
        self.blocks = nn.ModuleList([Block(start_channels * 2 ** i).cuda() for i in range(downsamplings)])
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=start_channels * img_size ** 2 // (2 ** downsamplings),
                                 out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=2 * latent_size)

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        mu, sigma = x[:, :x.shape[1] // 2], torch.exp(x[:, x.shape[1] // 2:])
        z = mu + sigma * torch.randn_like(mu)
        return z, (mu, sigma)


# Task 2

class Decoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, end_channels=16, upsamplings=5):
        super().__init__()
        self.last_conv = nn.Conv2d(in_channels=end_channels, out_channels=3, kernel_size=1,
                                   stride=1, padding=0)
        self.blocks = nn.ModuleList([Block(end_channels * (2 ** i), flag_up=True).cuda() for i in reversed(
            range(1, upsamplings + 1))])
        self.unflatten = nn.Unflatten(1, (end_channels * 2 ** upsamplings, img_size // (2 ** upsamplings),
                                          img_size // (2 ** upsamplings)))
        self.linear1 = nn.Linear(in_features=latent_size,
                                 out_features=256)
        self.linear2 = nn.Linear(in_features=256,
                                 out_features=end_channels * img_size ** 2 // (2 ** upsamplings))

    def forward(self, z):
        z = self.linear1(z)
        z = nn.ReLU()(z)
        z = self.linear2(z)
        z = self.unflatten(z)
        for block in self.blocks:
            z = block(z)
        z = self.last_conv(z)
        z = nn.Tanh()(z)
        return z


# Task 3

class VAE(nn.Module):
    def __init__(self, img_size=128, downsamplings=5, latent_size=128, down_channels=8, up_channels=13):
        super().__init__()
        self.encoder = Encoder(img_size=img_size, latent_size=latent_size, downsamplings=downsamplings,
                               start_channels=down_channels)
        self.decoder = Decoder(img_size=img_size, latent_size=latent_size, upsamplings=downsamplings,
                               end_channels=up_channels)

    def forward(self, x):
        z, (mu, sigma) = self.encoder(x)
        kld = 1 / 2 * (sigma ** 2 + mu ** 2 - torch.log(sigma ** 2) - 1)
        x_pred = self.decoder(z)
        return x_pred, kld

    def encode(self, x):
        z, (_, _) = self.encoder(x)
        return z

    def decode(self, z):
        x_pred = self.decoder(z)
        return x_pred

    def save(self):
        torch.save(self.state_dict(), 'model.pth')

    def load(self):
        self.load_state_dict(torch.load(__file__[:-7] + "model.pth"))
