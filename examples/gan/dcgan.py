import torch
from torch import nn

def DCGAN(h_ch=128, o_ch=3, f_h=4, f_w=4):
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()

            self.linear = nn.Linear(100, f_h*f_w*h_ch*8)
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(h_ch*8, h_ch*4, 5, 2, 2, 1),
                nn.InstanceNorm2d(h_ch*4),
                nn.Dropout2d(),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(h_ch*4, h_ch*2, 5, 2, 2, 1),
                nn.InstanceNorm2d(h_ch*2),
                nn.Dropout2d(),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(h_ch*2, h_ch, 5, 2, 2, 1),
                nn.InstanceNorm2d(h_ch),
                nn.Dropout2d(),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(h_ch, o_ch, 5, 2, 2, 1),
                nn.Tanh()
            )

        def forward(self, input_):
            output = self.linear(input_).view(-1, h_ch*8, f_h, f_w)
            output = self.conv(output)
            return output

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()

            self.conv = nn.Sequential(
                nn.Conv2d(o_ch, h_ch, 5, 2, 2),
                nn.Dropout2d(),
                nn.LeakyReLU(),
                nn.Conv2d(h_ch, h_ch*2, 5, 2, 2),
                nn.Dropout2d(),
                nn.LeakyReLU(),
                nn.Conv2d(h_ch*2, h_ch*4, 5, 2, 2),
                nn.Dropout2d(),
                nn.LeakyReLU(),
                nn.Conv2d(h_ch*4, h_ch*8, 5, 2, 2),
                nn.Dropout2d(),
                nn.LeakyReLU()
            )
            self.linear = nn.Sequential(
                nn.Linear(f_h*f_w*h_ch*8, 1),
                # nn.Sigmoid()
            )

        def forward(self, input_):
            output = self.conv(input_).view(-1, h_ch*8 * f_h * f_w)
            output = self.linear(output)
            return output

    return Generator(), Discriminator()

def main():
    # for testing
    from torch.autograd import Variable
    z = Variable(torch.randn(64, 100))
    g, d = DCGAN()
    x = g(z)
    print(x)
    print(d(x))


if __name__ == '__main__':
    main()