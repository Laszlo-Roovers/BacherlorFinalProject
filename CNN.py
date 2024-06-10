import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, name, channels):
        """Initialize components used to build the CNN model.

        Parameters
        ----------
        name : str
            Name used for storing results.
        channels : int
            Defines number of channels used in a layer of the CNN. Don't add input or output in parameter.
        """
        super().__init__()
        self.name = name

        # Construct the CNN using the given layout
        layers = []

        # Initially we only have the w and psi channel, at the end we output only w.
        channels = [2] + channels + [1]

        # Iteratively add layers
        for layer_index in range(len(channels) - 2):
            layers += [
                nn.Conv2d(
                    in_channels=channels[layer_index],
                    out_channels=channels[layer_index + 1],
                    kernel_size=5,
                    padding=2,
                    padding_mode="circular",
                ),
                nn.BatchNorm2d(channels[layer_index + 1]),
                nn.ReLU(),
            ]

        # In the last layer we only want a convolution without batch normalization and ReLU
        layers += [
            nn.Conv2d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=5,
                padding=2,
                padding_mode="circular",
            )
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """Calculate a forward pass.
    
        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x
