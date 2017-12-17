import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d()
        )
        self.linear = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 10),
            nn.Sigmoid()
        )

    def forward(self, input_):
        output = self.conv(input_)
        output = self.linear(output.view(-1, 64*7*7))
        return output


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.13066047740239478,), (0.3081078087569972,))
])

train_dataset = datasets.MNIST('../../data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)

test_dataset = datasets.MNIST('../../data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

def _var_to_numpy(var):
    return var.cpu().data.numpy()

cuda = torch.cuda.is_available()

conv_net = ConvNet()

if cuda:
    conv_net.cuda()

optimizer = torch.optim.Adam(conv_net.parameters(), lr=1e-3)
losses = []

def train(epoch):
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)

        if cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

        loss = F.cross_entropy(conv_net(batch_x), batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(_var_to_numpy(loss))

    print('Train epoch: {}, loss: {}'.format(epoch, _var_to_numpy(loss)))


for epoch in range(20):
    train(epoch)

conv_net.train(False)

correct = 0
for _, (batch_test_x , batch_test_y) in enumerate(test_loader):
    batch_test_x, batch_test_y = Variable(batch_test_x, volatile=True), Variable(batch_test_y)
    if cuda:
        batch_test_x = batch_test_x.cuda()
        batch_test_y = batch_test_y.cuda()

    correct += np.sum(_var_to_numpy(batch_test_y) == np.argmax(_var_to_numpy(conv_net(batch_test_x)), axis=1))

print(correct / len(test_dataset))

plt.plot(losses)
plt.show()