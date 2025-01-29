import torch
from torch import nn
import torch.nn.functional as F

from .downsample import Downsample
from .upsample import Upsample

class Discriminator(nn.Module):  
    def __init__(self):
        super(Discriminator, self).__init__()
      
        self.down1 = Downsample(6, 64) 
        self.down2 = Downsample(64, 128)  
        self.cov1 = nn.Conv2d(128, 256, kernel_size=3)  
        self.bn = nn.BatchNorm2d(256)
        self.last = nn.Conv2d(256, 1, kernel_size=3)  

    def forward(self, img, mask):  
        x = torch.cat([img, mask], dim=1)  
        x = self.down1(x, is_bn=False)  
        x = self.down2(x)
        x = F.dropout2d(self.bn(F.leaky_relu(self.cov1(x)))) 
        x = torch.sigmoid(self.last(x))
        return x