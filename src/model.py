import torch.nn as nn
from rnn import ConvLSTM, ConvLSTMCell, ConvRNN
from unet import UNet, UNetDecoder, UNetEncoder
from nl import NLUNet, NLUNetDecoder, NLUNetEncoder
from one_d import NLUNetEncoder1D, NLUNetDecoder1D, ConvLSTM1D


class UNetWrapper(nn.Module):
    """
    U-Net wrapper which flattens the batch and sequence dimensions allowing
    training on the same data input as sequence models.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, input_shape):
        super().__init__()
        self.unet = UNet(in_channels, out_channels, hidden_channels)
        self.nparams = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):  # BTcHW
        B, T, c, H, W = x.shape
        x = x.flatten(0, 1)  # (BT)cHW
        x = self.unet(x)
        x = x.view(B, T, -1, H, W)  # BTCHW
        return x


class NLUNetWrapper(nn.Module):
    """
    U-Net with NL operations wrapper which flattens the batch and sequence
    dimensions allowing training on the same data input as sequence models.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, input_shape):
        super().__init__()
        self.unet = NLUNet(in_channels, out_channels, hidden_channels)
        self.nparams = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):  # BTcHW
        B, T, c, H, W = x.shape
        x = x.flatten(0, 1)  # (BT)cHW
        x = self.unet(x)
        x = x.view(B, T, -1, H, W)  # BTCHW
        return x


class URNN(nn.Module):
    """
    UNet with a convolutional recurrent layer at the bottleneck.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, input_shape):
        super().__init__()
        self.encoder = UNetEncoder(in_channels, hidden_channels)
        self.decoder = UNetDecoder(hidden_channels, out_channels)
        self.rnn = ConvLSTM(hidden_channels, hidden_channels, input_shape)
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


class NLURNN(nn.Module):
    """
    UNet with non-local operation instead of 2nd convolution for each block.
    Also have a recurrent layer at the bottleneck.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, input_shape):
        super().__init__()
        self.encoder = NLUNetEncoder(in_channels, hidden_channels)
        self.decoder = NLUNetDecoder(hidden_channels, out_channels)
        self.rnn = ConvLSTM(hidden_channels, hidden_channels, input_shape)
        self.out_channels = out_channels
        self.nparams = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        B, T, c, H, W = x.shape
        x = x.flatten(0, 1)  # (BT)cHW
        x1, x2, x3, x4, x5 = self.encoder(x)
        _, C, h, w = x5.shape  # BTChw
        x5 = x5.view(B, T, C, h, w)
        x6, _ = self.rnn(x5)  # BTChw
        x5 = x5 + x6
        x = self.decoder(x1, x2, x3, x4, x5.view((-1, C, h, w)))
        return x.view(B, T, self.out_channels, H, W)


class NLURNNCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, input_shape):
        super().__init__()
        self.encoder = NLUNetEncoder(in_channels, hidden_channels)
        self.decoder = NLUNetDecoder(hidden_channels, out_channels)
        self.rnncell = ConvLSTMCell(hidden_channels, hidden_channels, input_shape)
        self.out_channels = out_channels
        self.nparams = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, hidden=None, cell=None):
        B, c, H, W = x.shape  # BcHW
        x1, x2, x3, x4, x5 = self.encoder(x)
        _, C, h, w = x5.shape  # BChw
        x6, cell = self.rnncell(x5, hidden, cell)  # BChw
        x5 = x5 + x6
        x = self.decoder(x1, x2, x3, x4, x5)  # B1HW
        return x, x5, cell


class NLURNN1D(nn.Module):
    """
    UNet with non-local operation instead of 2nd convolution for each block.
    Also have a recurrent layer at the bottleneck.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, input_shape):
        super().__init__()
        self.encoder = NLUNetEncoder1D(in_channels, hidden_channels)
        self.decoder = NLUNetDecoder1D(hidden_channels, out_channels)
        self.rnn = ConvLSTM1D(hidden_channels, hidden_channels, input_shape)
        self.out_channels = out_channels
        self.nparams = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        B, T, c, H, W = x.shape
        x = x.flatten(0, 1)  # (BT)cH1
        x1, x2, x3, x4, x5 = self.encoder(x)
        _, C, h, w = x5.shape  # BTCh1
        x5 = x5.view(B, T, C, h, w)
        x6, _ = self.rnn(x5)  # BTCh1
        x5 = x5 + x6
        x = self.decoder(x1, x2, x3, x4, x5.view((-1, C, h, w)))
        return x.view(B, T, self.out_channels, H, W)
