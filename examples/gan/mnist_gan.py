import argparse
import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from utils import var_to_numpy

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('-lr_d', type=float, default=2e-4)
parser.add_argument('-lr_g', type=float, default=2e-4)
args = parser.parse_args()

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input_):
        return self.linear(input_.view(-1, 784))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Sequential(
            nn.BatchNorm1d(100),
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, input_):
        return self.linear(input_).view(-1, 1, 28, 28)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.13066047740239478,), (0.3081078087569972,))
])

train_dataset = datasets.MNIST('../data',
                               train=True,
                               transform=transform,
                               download=True)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True)


d = Discriminator().cuda()
g = Generator().cuda()

optimizer_d = torch.optim.Adam(d.parameters(), lr=args.lr_d)
optimizer_g = torch.optim.Adam(g.parameters(), lr=args.lr_g)

criterion = nn.BCELoss()

def train(epoch):
    d.train()
    g.train()

    for i, (x, _) in enumerate(train_loader):
        z = Variable(torch.randn(len(x), 100)).cuda()
        x = Variable(x).cuda()

        real_labels = Variable(torch.ones(len(x))).cuda()
        fake_labels = Variable(torch.zeros(len(x))).cuda()

        loss_d = criterion(d(x), real_labels) + criterion(d(g(z)), fake_labels)

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        if i != 0 and i % 10 == 0:
            loss_g = criterion(d(g(z)), real_labels)

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

        if i != 0 and i % 100 == 0:
            print('Train epoch: {}, batch index: {}, loss_d: {}, loss_g: {}'.format(epoch, i, var_to_numpy(loss_d), var_to_numpy(loss_g)))


def plot(epoch):
    g.train(False)

    z = Variable(torch.randn(64, 100), volatile=True).cuda()
    fake_x = g(z).cpu()
    fake_x = (fake_x + 1) / 2
    grid = torchvision.utils.make_grid(fake_x.data)
    os.makedirs('images', exist_ok=True)
    torchvision.utils.save_image(grid, 'images/mnist_gan_{}.jpg'.format(epoch))

def main():
    for epoch in range(40):
        train(epoch)
        if epoch % 10 == 0:
            plot(epoch)


if __name__ == '__main__':
    main()
