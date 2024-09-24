import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.avgPool = nn.AvgPool2d(kernel_size=8, stride=0, padding=0, ceil_mode=False, count_include_pad=True)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.model(inputs)
        return x
    

class Decoder(nn.Module):
    def __init__(self, input_channels=128):
        super(Decoder, self).__init__()
        
        self.deconv1 = nn.Conv2d(input_channels//2, input_channels//2, kernel_size=7, padding='same', padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(input_channels//2)
        
        self.deconv2 = nn.Conv2d(input_channels//2, input_channels//4, kernel_size=5, padding='same', padding_mode='replicate')
        self.bn2 = nn.BatchNorm2d(input_channels//4)
        
        self.deconv3 = nn.Conv2d(input_channels//4, input_channels//8, kernel_size=5, padding='same', padding_mode='replicate')
        self.bn3 = nn.BatchNorm2d(input_channels//8)
        
        self.deconv4 = nn.Conv2d(input_channels//8, input_channels//16, kernel_size=5, padding='same', padding_mode='replicate')
        self.bn4 = nn.BatchNorm2d(input_channels//16)
        
        self.deconv5 = nn.Conv2d(input_channels//16, input_channels//32, kernel_size=3, padding='same', padding_mode='replicate')
        self.bn5 = nn.BatchNorm2d(input_channels//32)
        
        self.conv = nn.Conv2d(input_channels//32, 1, kernel_size=3, padding='same', padding_mode='replicate')

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        x = self.deconv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.deconv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        x = self.conv(x)
        x = self.sigmoid(x)
        
        return x