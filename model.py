from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    # build four-layer multi-layer perception
    def __init__(self, input_nodes, output_nodes, hidden_units=128):
        """
        :param input_nodes: the size of input units
        :param output_nodes: the size of output units
        :param hidden_units: the size of hidden units
        """
        super().__init__()
        self.fc1 = nn.Linear(input_nodes,hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, output_nodes)

    def forward(self, x):
        """
        Forward pass of the MLP
        :param x: the input
        :return x: output of forward pass of th model
        """
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

