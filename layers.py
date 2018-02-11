import torch
import torch.nn as nn


class HighwayMLP(nn.Module):

    def __init__(self,
                 input_size,
                 gate_bias=0,
                 activation_function=nn.functional.relu,
                 gate_activation=nn.functional.softmax):

        super(HighwayMLP, self).__init__()

        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.transfrom_layer = nn.Linear(input_size, input_size)

        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):

        T = self.activation_function(self.transfrom_layer(x))
        G = self.gate_activation(self.gate_layer(x))

        return T.mul(G) + x.mul(1-G)
