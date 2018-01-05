import argparse

# import matplotlib.pyplot as plt
import numpy as np
from time import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from utils import var_to_numpy

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--parallel', action='store_true')
args = parser.parse_args()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d()
        )
        self.linear = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.BatchNorm2d(1024),
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

train_dataset = datasets.MNIST('../data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

test_dataset = datasets.MNIST('../data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


net = Net()
if args.parallel:
    print('Use data parallel.')
    net = nn.DataParallel(net)

if torch.cuda.is_available():
    print('Use cuda.')
    net.cuda()


optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
losses = []

def train(epoch):
    net.train()

    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)

        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

        loss = F.cross_entropy(net(batch_x), batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(var_to_numpy(loss)))


    acc = evaluation()

    print('Train epoch: {}, loss: {}, acc: {}'.format(epoch, var_to_numpy(loss), acc))



def evaluation():
    net.eval()

    correct = 0
    for _, (batch_test_x , batch_test_y) in enumerate(test_loader):
        batch_test_x, batch_test_y = Variable(batch_test_x, volatile=True), Variable(batch_test_y)

        if torch.cuda.is_available():
            batch_test_x = batch_test_x.cuda()
            batch_test_y = batch_test_y.cuda()

        correct += np.sum(var_to_numpy(batch_test_y) == np.argmax(var_to_numpy(net(batch_test_x)), axis=1))

    return correct / len(test_dataset)

start = time()
for epoch in range(10):
    train(epoch)
end = time()

print('Time: {}'.format(end - start))

#plt.plot(losses)
#plt.show()
