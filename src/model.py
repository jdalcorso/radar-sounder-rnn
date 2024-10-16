import torch.nn as nn
from rnn import ConvLSTM
from unet import UNet, UNetDecoder, UNetEncoder


class UNetWrapper(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        n_layers,
        hidden_scaling,
        kernel_size,
    ):
        super().__init__()
        self.unet = UNet(in_channels, out_channels, hidden_channels)
        self.nparams = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):  # BTcHW
        B, T, c, H, W = x.shape
        x = x.flatten(0, 1)  # (BT)cHW
        x = self.unet(x)
        x = x.view(B, T, -1, H, W)  # BTCHW
        return x


class URNN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        n_layers,
        hidden_scaling,
        kernel_size,
    ):
        super().__init__()
        self.encoder = UNetEncoder(in_channels, hidden_channels)
        self.decoder = UNetDecoder(hidden_channels, out_channels)
        self.rnn = ConvLSTM(hidden_channels, hidden_channels)
        self.out_channels = out_channels
        self.nparams = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        B, T, c, H, W = x.shape
        x = x.flatten(0, 1)  # (BT)cHW
        x1, x2, x3, x4, x5 = self.encoder(x)
        _, C, h, w = x5.shape  # BTChw
        x5, _ = self.rnn(x5.view(B, T, C, h, w))  # BTChw
        x = self.decoder(x1, x2, x3, x4, x5.view((-1, C, h, w)))
        return x.view(B, T, self.out_channels, H, W)
