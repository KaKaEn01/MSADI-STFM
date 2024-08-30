import torch
import torch.nn as nn
from common import *

NUM_BANDS=6
channel_count=16


class UnetExtract(nn.Module):
    def __init__(self,in_channels=NUM_BANDS):
        super(UnetExtract,self).__init__()
       
        channels=(channel_count,channel_count*2,channel_count*4,channel_count*8,channel_count*16)
        
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, 1, 3), 
            ResBlock(channels[0]),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, 2, 1), 
            ResBlock(channels[1]),
        
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, 2, 1),
            ResBlock(channels[2]),
        
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, 2, 1), 
            ResBlock(channels[3]),
        
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], 3, 2, 1), 
            ResBlock(channels[4]),
        )

    
    def forward(self,inputs): 
        l1=self.conv1(inputs)
        l2=self.conv2(l1)
        l3=self.conv3(l2)
        l4=self.conv4(l3)
        l5=self.conv5(l4)
         
        return [l1,l2,l3,l4,l5]
    
    

class UnetDecoder(nn.Module):
    def __init__(self):
        super(UnetDecoder,self).__init__()

        
        self.up1=DeconvBlock(16*channel_count,8*channel_count,4,2,1,bias=True)
        self.up2=DeconvBlock(8*channel_count,4*channel_count,4,2,1,bias=True)
        self.up3=DeconvBlock(4*channel_count,2*channel_count,4,2,1,bias=True)
        self.up4=DeconvBlock(2*channel_count,channel_count,4,2,1,bias=True)
       
        self.conv_C=ResizeBlock(channel_count)
        self.conv_2C=ResizeBlock(2*channel_count)
        self.conv_4C=ResizeBlock(4*channel_count)
        self.conv_8C=ResizeBlock(8*channel_count)
        
        self.up_tranpose_8C=nn.ConvTranspose2d(8*channel_count,8*channel_count,8,2,3)
        self.up_tranpose_4C=nn.ConvTranspose2d(4*channel_count,4*channel_count,8,2,3)

        self.tail_23C_8C=nn.Conv2d(23*channel_count,8*channel_count,1,padding=0,bias=True)
        self.tail_19C_4C=nn.Conv2d(19*channel_count,4*channel_count,1,padding=0,bias=True)
        self.tail_17C_2C=nn.Conv2d(17*channel_count,2*channel_count,1,padding=0,bias=True)
        
        self.channel_func_C=CALayer(channel_count)
        self.channel_func_2C=CALayer(2*channel_count)
        self.channel_func_4C=CALayer(4*channel_count)
        self.channel_func_8C=CALayer(8*channel_count)

        
        self.Rail_Block=nn.Sequential(
            nn.Conv2d(channel_count,NUM_BANDS,7,1,3),
            nn.LeakyReLU(inplace=True)
        )

    
    def forward(self,inputs):
        
        l4_4=self.up1(inputs[4]) 
        l4_0=self.conv_C(inputs[0]) 
        l4_0=self.conv_C(l4_0) 
        l4_0=self.conv_C(l4_0)

        l4_1=self.conv_2C(inputs[1]) 
        l4_1=self.conv_2C(l4_1) 

        l4_2=self.conv_4C(inputs[2]) 

        l4_3=inputs[3] 
        
        l4=self.tail_23C_8C(torch.concat([l4_0,l4_1,l4_2,l4_3,l4_4],dim=1))
        l4=self.channel_func_8C(l4) 

        
        l3_4=self.up2(l4) 

        l3_0=self.conv_C(inputs[0]) 
        l3_0=self.conv_C(l3_0) 

        l3_1=self.conv_2C(inputs[1])
        
        l3_2=inputs[2] 

        l3_3=self.up_tranpose_8C(inputs[3]) 

        l3=self.tail_19C_4C(torch.concat([l3_0,l3_1,l3_2,l3_3,l3_4],dim=1))
        l3=self.channel_func_4C(l3) 

        
        l2_4=self.up3(l3) 

        l2_0=self.conv_C(inputs[0])
        
        l2_1=inputs[1] 
        
        l2_2=self.up_tranpose_4C(inputs[2])
        
        l2_3=self.up_tranpose_8C(inputs[3])
        l2_3=self.up_tranpose_8C(l2_3) 

        l2=self.tail_17C_2C(torch.concat([l2_0,l2_1,l2_2,l2_3,l2_4],dim=1))
        l2=self.channel_func_2C(l2) 

        l1=self.up4(l2) 
        
        output=self.Rail_Block(l1)

        return output
