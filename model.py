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

