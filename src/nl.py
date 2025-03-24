import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import OutConv


class NL1D(nn.Module):
    """Input should be BCHW, output is BCHW"""

    def __init__(self, out_channels, mid_channels):
        super().__init__()
        self.g = nn.Conv1d(out_channels, mid_channels, 1)
        self.outc = nn.Conv1d(mid_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)  # Add BN

    def forward(self, patch):  # BCHW
        _, C, _, W = patch.shape
        x = patch.mean(3)  # BCH
        x = F.layer_norm(x, x.shape[1:])  # Normalize more stably
        f = torch.bmm(x.transpose(1, 2), x) / (C**0.5)  # Scaled dot-product
        f = F.softmax(f, dim=2)  # BHH
        y = self.g(x).transpose(1, 2)  # BHc
        y = torch.bmm(f, y).transpose(1, 2)  # BcH
        y = self.outc(y).unsqueeze(-1)  # BCH1
        out = patch + y.repeat(1, 1, 1, W)  # Residual
        return F.silu(self.bn(out))  # Post-activation


class ConvNL(nn.Module):
    """(convolution => [BN] => ReLU) -> (nonlocal => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.nl = NL1D(out_channels, out_channels)  # TODO

    def forward(self, x):
        if x.shape[-1] == 1:  # to deal with w = 16
            x = x.repeat([1, 1, 1, 3])
        x = self.conv(x)
        return self.nl(x)


class DownNL(nn.Module):
    """Downscaling with maxpool then conv-nl"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), ConvNL(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpNL(nn.Module):
    """Upscaling then conv-nl"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = ConvNL(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class NLUNetEncoder(nn.Module):
    def __init__(self, n_channels=1, n_out=512):
        super().__init__()
        self.inc = ConvNL(n_channels, n_out // 16)
        self.down1 = DownNL(n_out // 16, n_out // 8)
        self.down2 = DownNL(n_out // 8, n_out // 4)
        self.down3 = DownNL(n_out // 4, n_out // 2)
        self.down4 = DownNL(n_out // 2, n_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class NLUNetDecoder(nn.Module):
    def __init__(self, n_channels=512, n_classes=4):
        super().__init__()
        self.up1 = UpNL(n_channels, n_channels // 2)
        self.up2 = UpNL(n_channels // 2, n_channels // 4)
        self.up3 = UpNL(n_channels // 4, n_channels // 8)
        self.up4 = UpNL(n_channels // 8, n_channels // 16)
        self.outc = OutConv(n_channels // 16, n_classes)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class NLUNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(NLUNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.encoder = NLUNetEncoder(in_channels, hidden_channels)
        self.decoder = NLUNetDecoder(hidden_channels, out_channels)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        x = self.decoder(x1, x2, x3, x4, x5)
        return x
