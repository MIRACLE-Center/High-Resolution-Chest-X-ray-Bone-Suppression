# sub-parts of the U-Net model
##子部件
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unet.deform_conv import th_batch_map_offsets, th_generate_grid

class double_conv(nn.Module):##两次卷积
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        ##conv的初始化
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            ##直接写了一个con的顺序
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            ##归一化
            nn.ReLU(inplace=True),##inplace就是要不要修改原对象
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            ##3 是kenel_size,他可以写成变的，（3,5） pading也可以 （1,2）

            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    ##不用写backward 他会自己生成

class deform_inconv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(deform_inconv,self).__init__()
        self.deform_conv=ConvOffset2D(in_ch)
        self.con=double_conv(in_ch,out_ch)
    def forward(self, x):
        deform_x=self.deform_conv(x)
        x=self.con(deform_x)
        return x


class inconv(nn.Module):##再封装了一次，注意，这里要继承Module
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            ##这个2还是kenel size
            double_conv(in_ch, out_ch)
        )
    def forward(self, x):
        x = self.mpconv(x)
        return x

class deform_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(deform_down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            ##这个2还是kenel size
            ConvOffset2D(in_ch),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:##当一个开关
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
                ##用的是反卷积
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        ##这里是拼接的过程，
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        ##找到差值，进行尺寸修正
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class deform_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(deform_up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:##当一个开关
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
                ##用的是反卷积
        self.conv = double_conv(in_ch, out_ch)
        self.deform_con=ConvOffset2D(in_ch)
    def forward(self, x1, x2):
        ##这里是拼接的过程，
        x1 = self.up(x1)

        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        ##找到差值，进行尺寸修正
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))

        x = torch.cat([x2, x1], dim=1)
        x = self.deform_con(x)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        ##kenel的size 是1*1
    def forward(self, x):
        x = self.conv(x)
        return x


class ConvOffset2D(nn.Conv2d):

    # 最关键的一点，他继承了conv2d

    """ConvOffset2D

    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation
    # 注意 这里说 他不做真正的convolution操作
    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """
    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init

        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.filters = filters
        # chanel数
        self._grid_param = None
        super(ConvOffset2D, self).__init__(self.filters, self.filters*2, 3, padding=1, bias=False, **kwargs)
        # 输出是输入chanel的两倍，因为是x和y两个偏移值，
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))
    #     给offset进行初始化，为0.01
    def forward(self, x):
        """Return the deformed featured map"""
        # x是上个层传下来的feature map
        x_shape = x.size()
        offsets = super(ConvOffset2D, self).forward(x)

        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)
        # 这里是坐标
        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h, w)
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self,x))
        # 这里是值
        # x_offset: (b, h, w, c)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)

        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x