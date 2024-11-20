import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import Down, DoubleConv, OutConv


class Gate(nn.Module):
    """
    Attention gate as per https://arxiv.org/abs/1804.03999,
    Official implementation: https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/layers/grid_attention_layer.py
    """

    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.theta = nn.Conv2d(
            in_channels=in_channels // 2,
            out_channels=mid_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )
        self.phi = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.psi = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.W = nn.Sequential(
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(mid_channels),
        )

    def forward(self, x, g):  # x:BcHW, g:BChw
        input_size = x.size()  # BcHW
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode="bilinear")
        f = F.relu(theta_x + phi_g, inplace=True)
        sigm_psi_f = F.sigmoid(self.psi(f))  # B1hw

        sigm_psi_f = F.interpolate(
            sigm_psi_f, size=input_size[2:], mode="bilinear"
        )  # B1HW
        y = sigm_psi_f.expand_as(x) * x  # BcHW
        W_y = self.W(y)  # BcHW

        return W_y


class UpGate(nn.Module):
    """
    Attention gate, Upscaling then double convolution.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)
        self.gate = Gate(in_channels, out_channels)

    def forward(self, x1, x2):
        # Gating
        x2 = self.gate(x2, x1)  # x,g

        # UpConv
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module with 4 levels of dilation (1,6,12,18).
    Dilated-Convolution maps are concatenated and processed by a final convolution
    """

    def __init__(self, in_channels=128, mid_channels=128):
        super().__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding="same",
                dilation=1,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding="same",
                dilation=6,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding="same",
                dilation=12,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding="same",
                dilation=18,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.aspp_out = nn.Sequential(
            nn.Conv2d(
                in_channels=mid_channels * 4,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding="same",
                dilation=1,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.aspp_out(x)
        return x


class UNetASPP(nn.Module):
    """
    Modified U-Net presented in https://ieeexplore.ieee.org/abstract/document/9605569
    with ASPP module at the bottleneck and Attention residual gates.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(UNetASPP, self).__init__()
        n_channels = in_channels
        n_classes = out_channels
        bilinear = False

        # Usual UNet
        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        factor = 2 if bilinear else 1
        self.down4 = Down(64, 128 // factor)
        self.up1 = UpGate(128, 64 // factor)
        self.up2 = UpGate(64, 32 // factor)
        self.up3 = UpGate(32, 16 // factor)
        self.up4 = UpGate(16, 8)
        self.outc = OutConv(8, n_classes)

        # ASPP
        self.aspp = ASPP()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.aspp(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetASPPWrapper(nn.Module):
    """
    Wrapper that merges batch and sequence dimension.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, input_shape):
        super().__init__()
        self.unet = UNetASPP(in_channels, hidden_channels, out_channels)
        self.nparams = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):  # BTcHW
        B, T, c, H, W = x.shape
        x = x.flatten(0, 1)  # (BT)cHW
        x = self.unet(x)
        x = x.view(B, T, -1, H, W)  # BTCHW
        return x
