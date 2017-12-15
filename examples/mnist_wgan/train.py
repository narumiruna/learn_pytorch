from model import Generator, Discriminator
from torchvision import transforms, datasets
import torch
from torch.autograd import Variable

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1))
])

train_dataset = datasets.MNIST( '../../data', transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size)


g = Generator()
d = Discriminator()

learning_rate = 0.00005
optimizer_g = torch.optim.RMSprop(g.parameters(), lr=learning_rate)
optimizer_d = torch.optim.RMSprop(d.parameters(), lr=learning_rate)


num_critic = 10
def train(epoch):
    for _, (real_x, _) in enumerate(train_loader):
        # train d
        real_x = Variable(real_x)
        for _ in range(num_critic):

            z = Variable(torch.randn(batch_size, 7*7))
            fake_x = g(z)
            loss_d = -d(real_x).mean() + d(fake_x).mean()

            optimizer_d.zero_grad()
            loss_d.backward()
            # clip
            for p in d.parameters():
                p.data.clamp_(-0.01, 0.01)
            optimizer_d.step()

        # train g
        z = Variable(torch.randn(batch_size, 7*7))
        fake_x = g(z)
        loss_g = -d(fake_x).mean()

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        print('discriminator loss: {}, generator loss: {}'.format(loss_d.data.numpy(), loss_g.data.numpy()))


def main():
    train(10)


if __name__ == '__main__':
    main()
