import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Here we have 1d versions of scripts from rnn.py, nl.py and unet.py
We put here a model (NLURNN1D) which operates on sequences
of rangelines (ablation study in the proposed paper).
"""


class NL1D1D(nn.Module):
    """Input should be BCHW, output is BCHW"""

    def __init__(self, out_channels, mid_channels):
        super().__init__()
        self.g = nn.Conv1d(out_channels, mid_channels, 1)
        self.outc = nn.Conv1d(mid_channels, out_channels, 1)
        pass

    def forward(self, patch):  # BCH1
        x = patch.squeeze(3)  # BCH
        x = F.normalize(x)
        f = torch.exp(torch.bmm(x.transpose(1, 2), x))  # BHH
        f = F.softmax(f, 2)  # BHH
        y = self.g(x).transpose(1, 2)  # BHc
        y = torch.bmm(f, y).transpose(1, 2)  # BcH
        y = self.outc(y).unsqueeze(-1)  # BCH1
        return patch + y


class ConvNL1D(nn.Module):
    """(convolution => [BN] => ReLU) -> (nonlocal => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.nl = NL1D1D(out_channels, out_channels)  # TODO

    def forward(self, x):
        x = self.conv(x.squeeze(-1))
        return self.nl(x.unsqueeze(-1))


class DownNL1D(nn.Module):
    """Downscaling with maxpool then conv-nl"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2), ConvNL1D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x.squeeze(-1))


class UpNL1D(nn.Module):
    """Upscaling then conv-nl"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = ConvNL1D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1.squeeze(-1))
        diffY = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2.squeeze(-1), x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x.squeeze(-1))


class NLUNetEncoder1D(nn.Module):
    def __init__(self, n_channels=1, n_out=512):
        super().__init__()
        self.inc = ConvNL1D(n_channels, n_out // 16)
        self.down1 = DownNL1D(n_out // 16, n_out // 8)
        self.down2 = DownNL1D(n_out // 8, n_out // 4)
        self.down3 = DownNL1D(n_out // 4, n_out // 2)
        self.down4 = DownNL1D(n_out // 2, n_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class NLUNetDecoder1D(nn.Module):
    def __init__(self, n_channels=512, n_classes=4):
        super().__init__()
        self.up1 = UpNL1D(n_channels, n_channels // 2)
        self.up2 = UpNL1D(n_channels // 2, n_channels // 4)
        self.up3 = UpNL1D(n_channels // 4, n_channels // 8)
        self.up4 = UpNL1D(n_channels // 8, n_channels // 16)
        self.outc = OutConv(n_channels // 16, n_classes)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ConvLSTM1D(nn.Module):
    def __init__(self, in_channels, out_channels, input_shape):
        super().__init__()
        H, W = input_shape[0] // 16, 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(
            in_channels + out_channels, out_channels * 4, kernel_size=3, padding="same"
        )
        self.Wci = nn.Parameter(torch.zeros(1, out_channels, H, 1))
        self.Wcf = nn.Parameter(torch.zeros(1, out_channels, H, 1))
        self.Wco = nn.Parameter(torch.zeros(1, out_channels, H, 1))

    def forward(self, x, h=None, c=None):
        B, T, _, H, _ = x.shape  # x is BTcHW
        if h is None:  # BCHW
            h = torch.rand((B, self.out_channels, H, 1), dtype=torch.float).to("cuda")

        if c is None:  # BCHW
            c = torch.rand((B, self.out_channels, H, 1), dtype=torch.float).to("cuda")

        h_final = []
        c_final = []
        for i in range(T):
            x_i = x[:, i, ...]  # x_i is BcH1
            cat_x = torch.cat([x_i, h], dim=1)  # B(c+C)H1
            conv_x = self.conv(cat_x.squeeze(-1)).unsqueeze(-1)  # B(4C)H1
            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)  # BCH1 * 4
            i = torch.sigmoid(i + self.Wci * c)
            f = torch.sigmoid(f + self.Wcf * c)
            c = f * c + i * torch.tanh(tmp_c)
            o = torch.sigmoid(o + self.Wco * c)
            h = o * torch.tanh(c)

            h_final.append(h.unsqueeze(1))  # B1CH1
            c_final.append(c.unsqueeze(1))  # B1CH1

        return torch.cat(h_final, dim=1), torch.cat(c_final, dim=1)  # BTCH1 * 2
