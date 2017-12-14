import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            Reshape(-1, 1, 7, 7),
            nn.ConvTranspose2d(1, 1024, 5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 1, 5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm2d(1, affine=True),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            Reshape(-1, 64 * 4 * 4),
            nn.Linear(64 * 4 * 4, 1)
        )

    def forward(self, x):
        return self.main(x)