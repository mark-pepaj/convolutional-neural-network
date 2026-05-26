import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

from model import CNN

# augmentation
random_rotate = transforms.RandomRotation(10) # 10 degrees
random_affine = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5)
horizontal_flip = transforms.RandomHorizontalFlip(p=0.5) # prob that it flips
vertical_flip = transforms.RandomVerticalFlip(p=0.5) # prob that it flips
augment_shape = transforms.RandomResizedCrop((28, 28), scale=(0.5, 1), ratio=(0.5, 2))
augment_color = transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0)

train_augments = transforms.Compose([
    random_affine,
    transforms.Pad(2),
    transforms.ToTensor()
    ])
val_augments = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])


train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True,
    transform=train_augments
)

# Load test data
val_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    transform=val_augments
)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


channels = [1, 6, 16, 120]
layer_configs = [
        {"type": "convolutional", "kernel_size": 5, "stride": 1, "padding": 0, "padding_mode": "reflect"},
        {"type": "avg_pooling", "kernel_size": 2, "stride": 2},
        {"type": "convolutional", "kernel_size": 5, "stride": 1, "padding": 0, "padding_mode": "reflect"},
        {"type": "avg_pooling", "kernel_size": 2, "stride": 2},
        {"type": "convolutional", "kernel_size": 5, "stride": 1, "padding": 0, "padding_mode": "reflect"},
        {"type": "flatten"},
        {"type": "linear", "in_features": 120, "out_features": 84},
        {"type": "nonlinearity", "name": "relu"},
        {"type": "linear", "in_features": 84, "out_features": 10},
        {"type": "nonlinearity", "name": "softmax"},
]

model = CNN(channels=channels, layer_configs=layer_configs).to(device)
criterion = model.get_criterion(layer_configs)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for X, target in train_loader:
        X, target = X.to(device), target.to(device)

        out = model(X)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for X, target in val_loader:
            X, target = X.to(device), target.to(device)

            out = model(X)
            val_loss += criterion(out, target).item()
            correct += (out.argmax(dim=1) == target).sum().item()

    print(f"Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss/len(val_loader):.4f}, val_acc={correct/len(val_dataset):.4f}")
