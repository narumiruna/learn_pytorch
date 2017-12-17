import argparse
import os

import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from dcgan import DCGAN

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-bs', type=int, default=128)
parser.add_argument('--learning_rate', '-lr', type=float, default=0.00005)
parser.add_argument('--clipping_parameter', '-cp', type=float, default=0.01)
parser.add_argument('--dir', '-d', type=str, default='fake_images')
parser.add_argument('--size', '-s', type=int, default=28)
args = parser.parse_args()


def _var_to_numpy(x):
    return x.cpu().data.numpy()

transform = transforms.Compose([
    transforms.Resize(args.size),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = datasets.MNIST('../data', train=True, transform=transform, download=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

g, d = DCGAN(h_ch=100,o_ch=1)

if torch.cuda.is_available():
    g.cuda()
    d.cuda()

# optimizer_d = torch.optim.Adam(d.parameters(), lr=args.learning_rate)
# optimizer_g = torch.optim.SGD(g.parameters(), lr=args.learning_rate)
optimizer_d = torch.optim.RMSprop(d.parameters(), lr=args.learning_rate)
optimizer_g = torch.optim.RMSprop(g.parameters(), lr=args.learning_rate)

logs = {'loss_d':[], 'loss_g':[]}

def train(epoch=0):
    for batch_idx, (x, _) in enumerate(dataloader):
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

        if batch_idx%5==0:
            loss_g = -d(g(z)).mean()
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

        logs['loss_d'].append(_var_to_numpy(loss_d))
        logs['loss_g'].append(_var_to_numpy(loss_g))


        if batch_idx % 100 == 0:
            print('Epoch {}, batch index: {}, loss_d: {}, loss_g: {}'.format(
                epoch, batch_idx,_var_to_numpy(loss_d), _var_to_numpy(loss_g)))


            os.makedirs(args.dir, exist_ok=True)
            grid = torchvision.utils.make_grid((g(z).cpu().data + 1) / 2)
            torchvision.utils.save_image(grid, '{}/{}.jpg'.format(args.dir, batch_idx))


for epoch in range(300):
    train(epoch)



plt.plot(logs['loss_d'], label='loss_d')
plt.plot(logs['loss_g'], label='loss_g')
plt.legend()
plt.savefig('loss.jpg')
