import numpy as np
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST('../../data', train=True, transform=transform, download=True)

x = torch.stack([data for data, _ in train_dataset])
print(x.size(), x.mean(), x.std())
