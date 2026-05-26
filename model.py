import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self, channels: list[int], layer_configs: list[dict]):
        super().__init__()

        nonlinearity_mapping = {
                "relu": nn.ReLU(),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid(),
                "leaky_relu": nn.LeakyReLU(),
                "softmax": nn.LogSoftmax(dim=1),
        }
        
        self.conv_layers = nn.ModuleList()
        conv_idx = 0

        for layer in layer_configs:
            if layer["type"] == "convolutional":
                self.conv_layers.append(nn.Conv2d(
                    channels[conv_idx],
                    channels[conv_idx + 1],
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                    padding_mode=layer["padding_mode"]
                ))
                conv_idx += 1
            elif layer["type"] == "max_pooling":
                self.conv_layers.append(nn.MaxPool2d(
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"]
                ))
            elif layer["type"] == "avg_pooling":
                self.conv_layers.append(nn.AvgPool2d(
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"]
                ))
            elif layer["type"] == "flatten":
                self.conv_layers.append(nn.Flatten())
            elif layer["type"] == "linear":
                self.conv_layers.append(nn.Linear(
                    layer["in_features"],
                    layer["out_features"]
                ))
            elif layer["type"] == "nonlinearity":
                self.conv_layers.append(nonlinearity_mapping[layer["name"]])


    def get_criterion(self, layer_configs):
        last_nonlinearity = next((layer["name"] for layer in reversed(layer_configs) if layer["type"] == "nonlinearity"), None)
        
        if last_nonlinearity == "softmax":
            return nn.NLLLoss()
        else:
            return nn.CrossEntropyLoss()


    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        
        return x





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


batch_size = 64

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

model = CNN(channels=channels, layer_configs=layer_configs)

for X, target in train_loader:
    out = model(X)
    print(out)
    break
