import torch
from torch.nn import Sequential, Linear, Dropout, ReLU, BatchNorm1d


class MLP(torch.nn.Module):

     def __init__(
         self,
         num_layers: int,
         dropout_rate: float,
         num_input_units: int,
         num_hidden_units: int
     ):

         super().__init__()

         self._layers = torch.nn.ModuleList()

         for _ in range(num_layers):
             self._layers.append(
                 self.createLayer(
                     dropout_rate,
                     num_input_units,
                     num_hidden_units
                 )
             )
             num_input_units = num_hidden_units

         self._out_layer = Sequential(Linear(num_hidden_units, 1))

     def forward(self, x):

         for layer in self._layers:
             x = layer(x)

         return self._out_layer(x)

     def createLayer(
         self,
         dropout_rate: float,
         num_input_units: int,
         num_output_units: int
     ):
         return Sequential(
             Dropout(dropout_rate),
             Linear(num_input_units, num_output_units),
             ReLU(),
             BatchNorm1d(num_output_units)
         )
