from model import Generator, Discriminator
from torchvision import transforms, datasets
import torch


batch_size = 64


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0), (1))
])

train_dataset = datasets.MNIST('../../data', transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size)



g = Generator()
d = Discriminator()


optimizer_g = torch.optim.RMSprop(g.parameters())
optimizer_d = torch.optim.RMSprop(d.parameters())

def train(epoch):
    for _, (train_x, _) in enumerate(train_loader):
        pass
