from torch import nn
import torch.nn.functional as F

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upconv_relu = nn.Sequential( 
            nn.ConvTranspose2d(in_channels, out_channels,  
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.LeakyReLU(inplace=True),  
        )
        self.bn = nn.BatchNorm2d(out_channels)

   
    def forward(self, x, is_drop=False):
        x = self.upconv_relu(x)  
        x = self.bn(x)  
        if is_drop:  
            x = F.dropout2d(x)
        return x