import torch
import torch.nn as nn
import torch.nn.functional as F

# ResBlock
class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size = 3, stride = 1, padding = 1, bias = True):
        super(ResBlock, self).__init__()
        residual = [
            nn.ReflectionPad2d(padding), 
            nn.Conv2d(in_channels, in_channels,kernel_size, stride, 0, bias=bias),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, in_channels,kernel_size, stride, 0, bias=bias),
            nn.Dropout(0.5), 
        ]
        self.residual = nn.Sequential(*residual)

    def forward(self, inputs):
        trunk = self.residual(inputs)
        return trunk + inputs
    
    


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.deconv(x)
        return self.act(out)
    
    

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
      
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x) 
        y = self.conv_du(y)
        return x * y


# 
class ResizeBlock(nn.Module):
    def __init__(self,in_channel,bias=True):
        super(ResizeBlock,self).__init__()
        self.maxPool=nn.MaxPool2d(2,2)
        self.conv1=nn.Conv2d(in_channel,in_channel,1,1,0)
        self.conv3=nn.Conv2d(in_channel,in_channel,3,1,1)
        self.leakyrelu=nn.LeakyReLU(inplace=True)
    def forward(self,x):
        x=self.maxPool(x)
        x=self.conv1(x)
        y=self.leakyrelu(self.conv3(x))
        return x+y
    