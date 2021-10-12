# full assembly of the sub-parts to form the complete net

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,no_deform=1):
        super(UNet, self).__init__()
        ##继承原来的init
        self.no_deform=no_deform
        self.inc = inconv(n_channels, 64)
        self.deform_inc=deform_inconv(n_channels,64)
        self.down1 = down(64, 128)
        self.deform_down1 = deform_down(64, 128)
        self.down2 = down(128, 256)
        self.deform_down2 = deform_down(128, 256)
        self.down3 = down(256, 512)
        self.deform_down3 = deform_down(256, 512)
        self.down4 = down(512, 512)
        self.deform_down4 = deform_down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.deform_up2=deform_up(512,128)
        self.up3 = up(256, 64)
        self.deform_up3=deform_up(256,64)
        self.up4 = up(128, 64)
        self.deform_up4=deform_up(128,64)
        self.outc = outconv(64, n_classes)
        self.convoffset64=ConvOffset2D(64)
        self.convoffset128=ConvOffset2D(128)
        self.convoffset256=ConvOffset2D(256)

    def forward(self, x):
        if self.no_deform:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            x = self.outc(x)
            return x
        else:
            x1 = self.inc(x)
            x2 = self.deform_down1(x1)
            x3 = self.deform_down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.deform_up3(x, x2)
            x = self.deform_up4(x,x1)
            x = self.outc(x)
            return x

