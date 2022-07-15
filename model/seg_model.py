import torch
from utils.common import initialize_weights
class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size, 
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class SegHead(torch.nn.Module):
    def __init__(self,backbone, num_lanes):
        super(SegHead, self).__init__()

        self.aux_header2 = torch.nn.Sequential(
            conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128,128,3,padding=1),
            conv_bn_relu(128,128,3,padding=1),
            conv_bn_relu(128,128,3,padding=1),
        )
        self.aux_header3 = torch.nn.Sequential(
            conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128,128,3,padding=1),
            conv_bn_relu(128,128,3,padding=1),
        )
        self.aux_header4 = torch.nn.Sequential(
            conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128,128,3,padding=1),
        )
        self.aux_combine = torch.nn.Sequential(
            conv_bn_relu(384, 256, 3,padding=2,dilation=2),
            conv_bn_relu(256, 128, 3,padding=2,dilation=2),
            conv_bn_relu(128, 128, 3,padding=2,dilation=2),
            conv_bn_relu(128, 128, 3,padding=4,dilation=4),
            torch.nn.Conv2d(128, num_lanes+1, 1)
            # output : n, num_of_lanes+1, h, w
        )

        initialize_weights(self.aux_header2,self.aux_header3,self.aux_header4,self.aux_combine)

        # self.droput = torch.nn.Dropout(0.1)
    def forward(self,x2,x3,fea):
        x2 = self.aux_header2(x2)
        x3 = self.aux_header3(x3)
        x3 = torch.nn.functional.interpolate(x3,scale_factor = 2,mode='bilinear')
        x4 = self.aux_header4(fea)
        x4 = torch.nn.functional.interpolate(x4,scale_factor = 4,mode='bilinear')
        aux_seg = torch.cat([x2,x3,x4],dim=1)
        aux_seg = self.aux_combine(aux_seg)
        return aux_seg