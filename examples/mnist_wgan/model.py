import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, in_features):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(in_features, 10 * 10)
        self.deconv1 = nn.ConvTranspose2d(1, 64, 3)
        self.unpool1 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 10, 10)
        x = self.deconv1(x)
        x = self.unpool1(x)
        return x


x = Variable(torch.Tensor(64, 100))
g = Generator(x.data.size()[-1])
print(g(x))
