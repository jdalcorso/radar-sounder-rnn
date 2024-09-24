import torch
import torch.nn as nn
import torch.nn.functional as f

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels + out_channels, out_channels * 4)

    def forward(self, x, h=None, cell=None):
        B, T, c, H, W = x.shape  # x is BTcHW
        if h is None: # BCHW
            h = torch.rand((B, self.out_channels, H,W), dtype=torch.float).to("cuda")

        if cell is None: # BCHW
            cell = torch.rand((B, self.out_channels,  H,W), dtype=torch.float).to("cuda")

        h_final = []
        c_final = []
        for i in range(T):
            x_i = x[:, i, ...]  # x_i is BcHW
            cat_x = torch.cat([x_i, h], dim=1)  # B(c+C)HW
            conv_x = self.conv(cat_x)  # B(4C)HW
            x_i, h_i, o_i, c_i = torch.chunk(conv_x, 4, dim=1)  # BCHW * 4

            x_i = torch.sigmoid(x_i)
            h_i = torch.sigmoid(h_i)
            o_i = torch.sigmoid(o_i)
            c_i = f.relu(c_i)

            c_out = (h_i * cell) + (x_i * cell)
            h_out = o_i * f.relu(cell)

            h_final.append(h_out.unsqueeze(1)) # B1CHW
            c_final.append(c_out.unsqueeze(1)) # B1CHW
            h = h_out
            cell = c_out
        return torch.cat(h_final, dim=1), torch.cat(c_final, dim=1)  # BTCHW * 2

class ConvRNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float))
        self.conv = nn.Conv2d(in_channels + out_channels, out_channels * 2)

    def forward(self, x, h=None):
        B, T, c, H, W = x.shape  # x is BTcHW
        if h is None:
            h = torch.rand((B, self.out_channels, H,W), dtype=torch.float).to(
                "cuda"
            )  # h is BCHW

        outputs = []
        for i in range(T):
            x_i = x[:, i, ...]  # x_i is BcHW
            cat_x = torch.cat([x_i, h], dim=1)  # B(c+C)HW
            conv_x = self.conv(cat_x)  # B(2C)HW
            x_i, h_i = torch.chunk(conv_x, 2, dim=1)  # BCHW, BCHW
            h_out = torch.tanh(x_i + h_i + self.bias.view(1, -1, 1))  # BCHW
            outputs.append(h_out.unsqueeze(1))  # B1CHW
            h_i = h_out
        return torch.cat(outputs, dim=1)  # BTCHW

