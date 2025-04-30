# This file contains the generator and critic models for the GAN.

import torch
import torch.nn as nn

# The generator takes a random noise vector as input and generates a 2D shape.
class Generator(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 2),
        )

    def forward(self, z):
        return self.net(z).view(-1, 64, 2)

# The critic takes a 2D shape as input and outputs a single score.
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64 * 2, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
