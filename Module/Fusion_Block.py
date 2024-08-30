import torch 
import torch.nn as nn
from common import *


class DM(nn.Module):
    def __init__(self, in_planes):
        super(DM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.max_pool = nn.AdaptiveMaxPool2d(1) 
        self.fc1 = nn.Conv2d(in_planes, in_planes//16, 1, bias=False) 
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Conv2d(in_planes//16, in_planes, 1, bias=False) 
        self.sigmoid = nn.Sigmoid()
    def forward(self, M0, M1): 
        diff = torch.sub(M1, M0)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(diff))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(diff))))
        att = self.sigmoid(avg_out + max_out)
        M0_feature = M0 * att + M0 
        M1_feature = M1 * att + M1 
        different = torch.sub(M1_feature, M0_feature)
        return M0_feature, M1_feature, different
    


class Fusion_Block(nn.Module):
    def __init__(self,in_channels):
        super(Fusion_Block,self).__init__()

        self.fusion=nn.Sequential(
            nn.Conv2d(in_channels*2,in_channels,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(in_channels)
        )
        self.DE=DM(in_channels) 
        self.sigmoid=nn.Sigmoid()
        self.convBlock=ResBlock(in_channels)

    def forward(self,M0_feature,M1_feature,F0_feature):
        
        M0_enhance,M1_enhance,Difference=self.DE(M0_feature,M1_feature) 
        M1_M0_cha_cat_F=self.fusion(torch.cat((Difference,F0_feature),dim=1)) # 
      
        W=self.sigmoid(M1_M0_cha_cat_F)
        F_out=W*M1_feature+(1-W)*F0_feature
        F_out=self.convBlock(F_out)
        
        return F_out
