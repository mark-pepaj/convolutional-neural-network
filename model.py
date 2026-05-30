import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConvolutionalLayer:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    padding_mode: str = "zeros"


@dataclass
class MaxPooling:
    kernel_size: int
    stride: int = 2


@dataclass
class AveragePooling:
    kernel_size: int
    stride: int = 2


@dataclass
class LinearLayer:
    in_features: int
    out_features: int


@dataclass
class Flatten:
    pass


@dataclass
class Nonlinearity:
    name: str  # "relu", "tanh", "sigmoid", "leaky_relu", "softmax"


@dataclass
class Criterion:
    name: str
    delta: float = 1.0
    beta: float = 1.0


@dataclass
class Optimizer:
    name: str
    lr: float = 1e-3
    momentum: float = 0.0
    weight_decay: float = 0.0


class CNN(nn.Module):
    def __init__(self, layer_configs: list):
        super().__init__()

        nonlinearity_mapping = {
                "relu": nn.ReLU(),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid(),
                "leaky_relu": nn.LeakyReLU(),
                "softmax": nn.LogSoftmax(dim=1),
        }
        
        self.conv_layers = nn.ModuleList()

        for config in layer_configs:
            if isinstance(config, ConvolutionalLayer):
                self.conv_layers.append(nn.Conv2d(
                    config.in_channels,
                    config.out_channels, 
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    padding=config.padding,
                    padding_mode=config.padding_mode
                ))
            elif isinstance(config, MaxPooling):
                self.conv_layers.append(nn.MaxPool2d(
                    kernel_size=config.kernel_size,
                    stride=config.stride
                ))
            elif isinstance(config, AveragePooling):
                self.conv_layers.append(nn.AvgPool2d(
                    kernel_size=config.kernel_size,
                    stride=config.stride
                ))
            elif isinstance(config, Flatten):
                self.conv_layers.append(nn.Flatten())
            elif isinstance(config, LinearLayer):
                self.conv_layers.append(nn.Linear(
                    config.in_features,
                    config.out_features
                ))
            elif isinstance(config, Nonlinearity):
                self.conv_layers.append(nonlinearity_mapping[config.name])
            else:
                raise ValueError(f"Unrecognized layer config type: {type(config)}")


    def get_criterion(self, layer_configs, criterion_config: Criterion = None):
        if criterion_config is not None:
            criterion_mapping = {
                    "cross_entropy":   nn.CrossEntropyLoss(),
                    "nll":             nn.NLLLoss(),
                    "bce":             nn.BCELoss(),
                    "bce_with_logits": nn.BCEWithLogitsLoss(),
                    "mse":             nn.MSELoss(),
                    "mae":             nn.L1Loss(),
                    "huber":           nn.HuberLoss(delta=criterion_config.delta),
                    "smooth_l1":       nn.SmoothL1Loss(beta=criterion_config.beta),
            }
            return criterion_mapping[criterion_config.name]

        last_nonlinearity = next((config.name for config in reversed(layer_configs) if isinstance(config, Nonlinearity)), None)
        

        inferred_mapping = {
                "softmax": nn.NLLLoss(),
                "sigmoid": nn.BCELoss(),
                None:      nn.CrossEntropyLoss(),
        }
        
        return inferred_mapping.get(last_nonlinearity, nn.CrossEntropyLoss())



    def get_optimizer(self, optimizer_config: Optimizer = None):

        if optimizer_config is None:
            return torch.optim.AdamW(self.parameters(), lr=1e-3)
        
        optimizer_mapping = {
                "Adam":    optim.Adam,
                "SGD":     optim.SGD,
                "RMSprop": optim.RMSprop,
                "AdamW":   optim.AdamW,
                "Adagrad": optim.Adagrad,
        }

        optimizer_cls = optimizer_mapping[optimizer_config.name]

        kwargs = {k: v for k, v in vars(optimizer_config).items() if k != "name" and v != 0.0}

        return optimizer_cls(self.parameters(), **kwargs) 



    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x



    def fit(self, train_loader, val_loader, num_epochs, layer_configs, criterion_config, optimizer_config, save_path="trained_weights.pth"):

        criterion = self.get_criterion(layer_configs, criterion_config)
        optimizer = self.get_optimizer(optimizer_config)

        device = next(self.parameters()).device

        history = {
                "train_loss": [],
                "val_loss": [],
                "val_acc": []
        }

        best_val_loss = float("inf")


        for epoch in range(num_epochs):
            self.train()
            train_loss = 0
            
            for X, target in train_loader:
                X, target = X.to(device), target.to(device)

                out = self(X)
                loss = criterion(out, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            self.eval()
            val_loss = 0
            correct = 0

            with torch.no_grad():
                for X, target in val_loader:
                    X, target = X.to(device), target.to(device)

                    out = self(X)

                    val_loss += criterion(out, target).item()
                    correct += (out.argmax(dim=1) == target).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = correct / len(val_loader.dataset)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(avg_val_acc)
            
            if save_path is not None and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.state_dict(), save_path)
                 

            print(f"Epoch {epoch+1}/{num_epochs}: TrainLoss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_acc={avg_val_acc:.4f}")
        
        return history
