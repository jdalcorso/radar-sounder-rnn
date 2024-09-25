import torch.nn as nn
from cnn import Encoder, Decoder
from rnn import ConvLSTM, ConvRNN
from unet import UNet

class Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, hidden_scaling, kernel_size):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, n_layers, hidden_scaling, kernel_size)
        self.decoder = Decoder(hidden_channels, out_channels, n_layers, hidden_scaling, kernel_size)
        self.rnn = ConvLSTM(hidden_channels, hidden_channels)
        self.out_channels = out_channels
        self.nparams = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x): # BTcHW
        outputs = []
        B,T,c,H,W = x.shape # c=1
        x_enc = self.encoder(x.view(-1,c,H,W)) # (BT)Chw
        _,C,h,w = x_enc.shape # BTChw
        x_enc = x_enc.view(B,T,C,h,w) # BTChw
        h_out, _ = self.rnn(x_enc) # BTChw
        x_dec = self.decoder(h_out.view(-1,C,h,w)) #(BT)1HW
        x_dec = x_dec.view(B,T,self.out_channels,H,W) # BTcHW
        return x_dec
    
class UNetWrapper(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, hidden_scaling, kernel_size):
        super().__init__()
        self.unet = UNet(in_channels,out_channels)
        self.nparams = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self,x):  # BTcHW
        B,T,c,H,W=x.shape
        x = x.flatten(0,1) #(BT)cHW
        x = self.unet(x)
        x = x.view(B,T,-1,H,W) # BTCHW
        return x