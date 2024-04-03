import torch
from torch.nn import BatchNorm1d, Dropout, Linear, ReLU, Sequential


class MLP(torch.nn.Module):
    """
    Create a multi layered pecepron model using pytorch
    """

    def __init__(
        self,
        num_layers: int,
        dropout_rate: float,
        num_input_units: int,
        num_hidden_units: int,
    ):
        super().__init__()

        self._layers = torch.nn.ModuleList()

        for _ in range(num_layers):
            self._layers.append(
                self.createLayer(dropout_rate, num_input_units, num_hidden_units)
            )
            num_input_units = num_hidden_units

        self._out_layer = Sequential(Linear(num_hidden_units, 1))

    def forward(self, x):
        """
        Iterate through the model to obtain a prediction.

        :param x: input data
        """

        for layer in self._layers:
            x = layer(x)

        return self._out_layer(x)

    def createLayer(
        self, dropout_rate: float, num_input_units: int, num_output_units: int
    ):
        """
        Create one layer of the MLP model consisting of dropout, linear, ReLU
        activation and batch normalization.

        :param dropout_rate: fraction of units that needs to be disabled during
            training.
        :param num_input_units: dimension of input vector
        :param num_output_units: dimension of output vector
        """

        return Sequential(
            Dropout(dropout_rate),
            Linear(num_input_units, num_output_units),
            ReLU(),
            BatchNorm1d(num_output_units),
        )
