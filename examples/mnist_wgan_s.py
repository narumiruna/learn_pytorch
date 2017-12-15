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
parser.add_argument('--size', '-s', type=int, default=28)
args = parser.parse_args()


class Generator(nn.Module):
    def __init__(self, num_hiddens):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, args.size**2),
            nn.Tanh()
        )

    def forward(self, input_):
        output = self.main(input_)
        return output.view(-1, 1, args.size, args.size)


class Discriminator(nn.Module):
    def __init__(self, num_hiddens):
        super(Discriminator, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(args.size**2, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, 1)
        )

    def forward(self, input_):
        output = self.linear(input_.view(-1, args.size**2))
        return output

transform = transforms.Compose([
    transforms.Resize(args.size),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataloader = torch.utils.data.DataLoader(datasets.MNIST(
    '../data', train=True, transform=transform, download=True), batch_size=args.batch_size, shuffle=True)

g = Generator(1000)
d = Discriminator(1000)
g.cuda()
d.cuda()

optimizer_d = torch.optim.RMSprop(d.parameters(), lr=args.learning_rate)
optimizer_g = torch.optim.RMSprop(g.parameters(), lr=args.learning_rate)


def train(epoch=0):
    for i, (x, _) in enumerate(dataloader):
        x = Variable(x).cuda()
        z = Variable(torch.randn(x.data.size()[0], 100)).cuda()

        loss_d = -d(x).mean() + d(g(z)).mean()
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()
        for w in d.parameters():
            w.data.clamp_(-args.clipping_parameter, args.clipping_parameter)

        loss_g = -d(g(z)).mean()
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        if i % 100 == 0:
            print('Epoch {}, step: {}, loss_d: {}, loss_g: {}'.format(
                epoch, i, loss_d.cpu().data.numpy(), loss_g.cpu().data.numpy()))

            os.makedirs(args.dir, exist_ok=True)
            z = Variable(torch.randn(64, 100), volatile=True).cuda()
            grid = torchvision.utils.make_grid((g(z).data + 1) / 2)
            torchvision.utils.save_image(grid, '{}/{}.jpg'.format(args.dir, i))

for epoch in range(1000):
    train(epoch)
