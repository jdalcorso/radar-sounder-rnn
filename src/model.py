import torch.nn as nn
from cnn import Encoder, Decoder
from rnn import ConvLSTM, ConvRNN

class Model(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 5):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.rnn = ConvLSTM(in_channels=128, out_channels=128)
        self.out_channels = out_channels

    def forward(self, x): # BTcHW
        outputs = []
        B,T,c,H,W = x.shape
        x_enc = self.encoder(x.view(-1,c,H,W)) # (BT)Chw
        _,C,h,w = x_enc.shape # BTChw
        x_enc = x_enc.view(B,T,C,h,w) # BTChw
        h_out, _ = self.rnn(x_enc) # BTChw
        x_dec = self.decoder(h_out.view(-1,C,h,w)) #(BT)1HW
        x_dec = x_dec.view(B,T,self.out_channels,H,W) # BTcHW
        return x_dec