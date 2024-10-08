import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, n_layers, hidden_scaling, kernel_size
    ):
        super().__init__()

        layers = []
        this_in_channels = in_channels
        for i in range(n_layers):
            this_out_channels = out_channels // (2 ** (n_layers - i - 1))
            layers.append(
                nn.Conv2d(
                    this_in_channels,
                    this_out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                )
            )
            layers.append(nn.BatchNorm2d(this_out_channels))
            layers.append(nn.ReLU())
            if hidden_scaling > 0:
                layers.append(nn.AvgPool2d(2))
                hidden_scaling -= 1
            this_in_channels = this_out_channels
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, n_layers, hidden_scaling, kernel_size
    ):
        super(Decoder, self).__init__()
        layers = []
        this_in_channels = in_channels
        for i in range(n_layers):
            this_out_channels = (
                in_channels // (2 ** (i + 1)) if i < n_layers - 1 else out_channels
            )
            if hidden_scaling > 0:
                layers.append(
                    nn.ConvTranspose2d(
                        this_in_channels,
                        this_out_channels,
                        kernel_size,
                        2,
                        kernel_size // 2,
                        1,
                    )
                )
                hidden_scaling -= 1
            else:
                layers.append(
                    nn.Conv2d(
                        this_in_channels,
                        this_out_channels,
                        kernel_size=kernel_size,
                        padding="same",
                    )
                )
            if i < n_layers - 1:
                layers.append(nn.BatchNorm2d(this_out_channels))
                layers.append(nn.ReLU())
            this_in_channels = this_out_channels
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x
