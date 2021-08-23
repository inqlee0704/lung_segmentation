# models
import torch
import torch.nn as nn
from torchsummary import summary
from RecursiveUNet3D import UNet3D

def double_conv(in_c, out_c, kernel_size=3, norm_layer=nn.BatchNorm3d):
    conv = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=kernel_size, padding=1),
        norm_layer(out_c),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_c, out_c, kernel_size=kernel_size, padding=1),
        norm_layer(out_c),
        nn.ReLU(inplace=True),     
    )
    return conv

def last_conv(in_c=4, out_c=8, num_c=2, kernel_size=3, norm_layer=nn.BatchNorm3d):
    conv = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=kernel_size, padding=1),
        norm_layer(out_c),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_c, num_c, kernel_size=kernel_size, padding=1),
        # nn.Softmax(dim=1)

    )
    return conv
        
class Seg3D(nn.Module):
    def __init__(self, num_classes=2, c_size=1):
        super(Seg3D, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.down_conv_1 = double_conv(c_size, 32)
        self.conv_1d_1 = nn.Conv3d(32,1,kernel_size=1)
        self.down_conv_2 = double_conv(32, 64)
        self.conv_1d_2 = nn.Conv3d(64,1,kernel_size=1)
        self.down_conv_3 = double_conv(64, 128)
        self.conv_1d_3 = nn.Conv3d(128,1,kernel_size=1)
        self.down_conv_4 = double_conv(128, 256)
        self.conv_1d_4 = nn.Conv3d(256,1,kernel_size=1)
        
        # Nearest neighbor interpolation 
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_2 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample_3 = nn.Upsample(scale_factor=8, mode='nearest')

        # Final
        self.conv_map = last_conv(in_c=4,out_c=8,num_c=num_classes)
    
    def forward(self,image):
        # encoder
        x1 = self.down_conv_1(image) 
        x1_1 = self.conv_1d_1(x1)

        x2 = self.max_pool(x1)

        x3 = self.down_conv_2(x2) 
        x3_1 = self.conv_1d_2(x3)
        x3_up = self.upsample_1(x3_1)

        x4 = self.max_pool(x3)

        x5 = self.down_conv_3(x4)
        x5_1 = self.conv_1d_3(x5)
        x5_up = self.upsample_2(x5_1)

        x6 = self.max_pool(x5)

        x7 = self.down_conv_4(x6) 
        x7_1 = self.conv_1d_4(x7)
        x7_up = self.upsample_3(x7_1)

        combined = torch.cat([x1_1,x3_up,x5_up,x7_up],1)
        out = self.conv_map(combined)

        return out