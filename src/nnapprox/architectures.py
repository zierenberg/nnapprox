import torch
import torch.nn as nn

class DenseNN(nn.Module):
    """
    This class implements a fully connected neural network
    with a variable number of hidden layers.

    Example
    -------
    >> model = DenseNN(input_size=3, hidden_sizes=[42,42,42], output_size=2)
    >> model(torch.tensor([1,2,3], dtype=torch.float32))

    Parameters
    ----------
    input_size: int
        Number of input neurons.
    hidden_sizes: list
        List of hidden layer sizes.
    output_size: int
        Number of output neurons.
    act: torch.nn.Module
        Activation function (default nn.Tanh()).
    """

    def __init__(self, input_size, hidden_sizes, output_size, act=nn.Tanh()):
        super().__init__()
        # create activation function
        self.act = act
        self.input_size = input_size
        self.output_size = output_size
        # create layers
        self.hidden = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        for i in range(1, len(hidden_sizes)):
            self.hidden.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        self.output = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.hidden:
            x = self.act(layer(x))

        return self.output(x)

