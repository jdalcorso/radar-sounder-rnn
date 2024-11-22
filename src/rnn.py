import torch
import torch.nn as nn
import torch.nn.functional as f

""" 
ConvLSTM is refactored from https://github.com/Hzzone/Precipitation-Nowcasting.
ConvRNN is custom made.
"""


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, input_shape):
        super().__init__()
        H, W = input_shape[0] // 16, input_shape[1] // 16
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels + out_channels, out_channels * 4, kernel_size=3, padding="same"
        )
        self.Wci = nn.Parameter(torch.zeros(1, out_channels, H, W))
        self.Wcf = nn.Parameter(torch.zeros(1, out_channels, H, W))
        self.Wco = nn.Parameter(torch.zeros(1, out_channels, H, W))

    def forward(self, x, h=None, c=None):
        B, T, _, H, W = x.shape  # x is BTcHW
        if h is None:  # BCHW
            h = torch.rand((B, self.out_channels, H, W), dtype=torch.float).to("cuda")

        if c is None:  # BCHW
            c = torch.rand((B, self.out_channels, H, W), dtype=torch.float).to("cuda")

        h_final = []
        c_final = []
        for i in range(T):
            x_i = x[:, i, ...]  # x_i is BcHW
            cat_x = torch.cat([x_i, h], dim=1)  # B(c+C)HW
            conv_x = self.conv(cat_x)  # B(4C)HW
            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)  # BCHW * 4
            i = torch.sigmoid(i + self.Wci * c)
            f = torch.sigmoid(f + self.Wcf * c)
            c = f * c + i * torch.tanh(tmp_c)
            o = torch.sigmoid(o + self.Wco * c)
            h = o * torch.tanh(c)

            h_final.append(h.unsqueeze(1))  # B1CHW
            c_final.append(c.unsqueeze(1))  # B1CHW

        return torch.cat(h_final, dim=1), torch.cat(c_final, dim=1)  # BTCHW * 2


class ConvRNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels + out_channels, out_channels * 2, kernel_size=3, padding="same"
        )

    def forward(self, x, h=None):
        B, T, c, H, W = x.shape  # x is BTcHW
        if h is None:
            h = torch.rand((B, self.out_channels, H, W), dtype=torch.float).to(
                "cuda"
            )  # h is BCHW

        outputs = []
        for i in range(T):
            x_i = x[:, i, ...]  # x_i is BcHW
            cat_x = torch.cat([x_i, h], dim=1)  # B(c+C)HW
            conv_x = self.conv(cat_x)  # B(2C)HW
            x_i, h_i = torch.chunk(conv_x, 2, dim=1)  # BCHW, BCHW
            h_out = torch.tanh(x_i + h_i)  # BCHW
            outputs.append(h_out.unsqueeze(1))  # B1CHW
            h_i = h_out
        return torch.cat(outputs, dim=1), 0  # BTCHW


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, input_shape):
        super().__init__()
        H, W = input_shape[0] // 16, input_shape[1] // 16
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels + out_channels,
            out_channels * 4,
            kernel_size=3,
            padding="same",
            padding_mode="reflect",
        )
        self.Wci = nn.Parameter(torch.randn(1, out_channels, H, W))
        self.Wcf = nn.Parameter(torch.randn(1, out_channels, H, W))
        self.Wco = nn.Parameter(torch.randn(1, out_channels, H, W))

    def forward(self, x, h=None, c=None):
        B, _, H, W = x.shape  # x is BcHW
        if h is None:  # BCHW
            h = torch.zeros((B, self.out_channels, H, W), dtype=torch.float).to("cuda")

        if c is None:  # BCHW
            c = torch.zeros((B, self.out_channels, H, W), dtype=torch.float).to("cuda")

        cat_x = torch.cat([x, h], dim=1)  # B(c+C)HW
        conv_x = self.conv(cat_x)  # B(4C)HW
        i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)  # BCHW * 4
        i = torch.sigmoid(i + self.Wci * c)
        f = torch.sigmoid(f + self.Wcf * c)
        c = f * c + i * torch.tanh(tmp_c)
        o = torch.sigmoid(o + self.Wco * c)
        h = o * torch.tanh(c)
        return h, c  # BCHW * 2