import torch
import torch.nn as nn
import torch.nn.functional as F


"""
class CNN(nn.module):
    def __init__(self, input_size, convolutional_layers_filters, pooling_layers, hidden_layers, num_output_neurons, hidden_activations=None, output_activation=None):
        super().__init__()

        if isinstance(hidden_activations, nn.Module):
            activations = [hidden_activations] * len(hidden_layers)

        assert len(hidden_activations) == len(hidden_layers), (f"activations length ({len(hidden_activations)}) must match hidden_layers_sizes ({len(hidden_layers)})")
    
        layer_sizes = [input_size] + hidden_layers
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(hidden_layers))
        ])
        self.hidden_activations = nn.ModuleList(hidden_activations)
        self.output_layer = nn.Linear(hidden_layers[-1], num_output_neurons)
        self.output_activation = output_activation


    def _convolution(self, x, kernel, stride):
        a b c d e f
        g h i j k l
        m n o p q r
        t s u v w x

        i = 0
        j = 0
        filter_size = kernel.shape[0]

        for i, j in range(input_size**0.5):




    def forward(self, x):
        
        for convolutional_layer in convolutional_layers:


        for hidden_layer, hidden_activation in zip(self.hidden_layers, self.hidden_activations):
            x = hidden_act(hidden_layer(x))

        x = self.output_layer(x)

        if self.output_activation is not None:
            x = self.output_activation(x)

        return x

"""



X = torch.abs(torch.randn(1, 12, 12))
X /= 255
print(X.shape)

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
result = conv(X)
print(result.shape)
