import argparse

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--log_interval', type=int, default=100)
args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()

print(args)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(32, 64, 5, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d()
        )

        self.linear = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 10),
        )

    def forward(self, input_):
        output = self.conv(input_)
        output = output.view(-1, 64 * 4 * 4)
        output = self.linear(output)
        return output


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.13066047740239478,), (0.3081078087569972,))
])

train_dataset = datasets.MNIST('data',
                               train=True,
                               transform=transform,
                               download=True)
train_loader = data.DataLoader(train_dataset,
                               batch_size=args.batch_size)
test_dataset = datasets.MNIST('data',
                              train=False,
                              transform=transform,
                              download=True)
test_loader = data.DataLoader(test_dataset,
                              batch_size=args.batch_size)

net = Net()

if args.cuda:
    net.cuda()

if args.parallel:
    net = nn.DataParallel(net)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
cross_entropy = nn.CrossEntropyLoss()

def train():
    net.train()
    for train_index, (train_x, train_y) in enumerate(train_loader):
        train_x = Variable(train_x)
        train_y = Variable(train_y)

        if args.cuda:
            train_x = train_x.cuda()
            train_y = train_y.cuda()

        loss = cross_entropy(net(train_x), train_y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if train_index % args.log_interval == 0:
            print('Batch index {}, loss: {}'.format(train_index, float(loss.data)))

def evaluate():
    net.eval()

    correct = 0
    for _, (test_x, test_y) in enumerate(test_loader):
        test_x = Variable(test_x, volatile=True)

        if args.cuda:
            test_x = test_x.cuda()
            test_y = test_y.cuda()

        _, max_indices = net(test_x).data.max(dim=1)
        correct += int((max_indices == test_y).sum())

    accuracy = correct / len(test_loader.dataset)

    print('Accuracy: {}'.format(accuracy))


for epoch in range(10):
    print('Train epoch: {}'.format(epoch))
    train()
    evaluate()
