import argparse
import os

import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

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


def _var_to_numpy(x):
    return x.cpu().data.numpy()

transform = transforms.Compose([
    transforms.Resize(args.size),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = datasets.MNIST('../data', train=True, transform=transform, download=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

g = Generator(200)
d = Discriminator(200)

if torch.cuda.is_available():
    g.cuda()
    d.cuda()

optimizer_d = torch.optim.RMSprop(d.parameters(), lr=args.learning_rate)
optimizer_g = torch.optim.RMSprop(g.parameters(), lr=args.learning_rate)

logs = {'loss_d':[], 'loss_g':[]}

def train(epoch=0):
    for i, (x, _) in enumerate(dataloader):
        x = Variable(x)
        z = Variable(torch.randn(x.data.size()[0], 100))

        if torch.cuda.is_available():
            x = x.cuda()
            z = z.cuda()

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

        logs['loss_d'].append(_var_to_numpy(loss_d))
        logs['loss_g'].append(_var_to_numpy(loss_g))


        if i % 100 == 0:
            print('Epoch {}, step: {}, loss_d: {}, loss_g: {}'.format(
                epoch, i,_var_to_numpy(loss_d), _var_to_numpy(loss_g)))


            os.makedirs(args.dir, exist_ok=True)
            grid = torchvision.utils.make_grid((g(z).cpu().data + 1) / 2)
            torchvision.utils.save_image(grid, '{}/{}.jpg'.format(args.dir, i))


for epoch in range(10):
    train(epoch)


plt.plot(logs['loss_d'], label='loss_d')
plt.plot(logs['loss_g'], label='loss_g')
plt.legend()
plt.savefig('loss.jpg')