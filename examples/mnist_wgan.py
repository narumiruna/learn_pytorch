import argparse
import os

import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-bs', type=int, default=64)
parser.add_argument('--learning_rate', '-lr', type=float, default=0.00005)
parser.add_argument('--clipping_parameter', '-cp', type=float, default=0.01)
parser.add_argument('--dir', '-d', type=str, default='fake_images')
args = parser.parse_args()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(4, 1024, 1, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input_):
        output = self.main(input_.view(-1, 4, 5, 5))
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 4, stride=2),
            nn.LeakyReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(32*5*5, 1),
        )

    def forward(self, input_):
        output = self.conv(input_)
        output = output.view(-1, 32*5*5)
        output = self.linear(output)
        return output


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1))
])

train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, transform=transform, download=True), batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transform, download=True), batch_size=args.batch_size)

g = Generator()
d = Discriminator()

optimizer_d = torch.optim.RMSprop(d.parameters(), lr=args.learning_rate)
optimizer_g = torch.optim.RMSprop(g.parameters(), lr=args.learning_rate)

def plot(step):
    os.makedirs(args.dir, exist_ok=True)
    z = Variable(torch.randn(64, 100), volatile=True)
    fake_x = (g(z).data + 1) / 2
    grid = torchvision.utils.make_grid(fake_x)
    torchvision.utils.save_image(grid,'{}/{}.jpg'.format(args.dir,step))


def train(epoch=0):
    for i, (x, _) in enumerate(train_loader):
        x = Variable(x)
        z = Variable(torch.randn(x.data.size()[0], 100))

        loss_d = -d(x).mean() + d(g(z)).mean()
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()
        for w in d.parameters():
            w.data.clamp_(-args.clipping_parameter, args.clipping_parameter)

        if i%10==0 and i!=0:
            loss_g = -d(g(z)).mean()
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

        if i%100 == 0:
            loss_d = -d(x).mean() + d(g(z)).mean()
            loss_g = -d(g(z)).mean()
            print('step: {}, loss_d: {}, loss_g: {}'.format(i, loss_d.data.numpy(), loss_g.data.numpy()))
            plot(i)

for epoch in range(20):
    train(epoch)
