from __future__ import absolute_import, division

import torch
from torch.autograd import Variable

import numpy as np
from scipy.ndimage.interpolation import map_coordinates as sp_map_coordinates
# 使用的三次样条插值，对x做一次，y做一次，然后对x和y的结果 做一次
#    th是torch的缩写
def th_flatten(a):
    """Flatten tensor"""
    return a.contiguous().view(a.nelement())
#     a的元素数

def th_repeat(a, repeats, axis=0):
    """Torch version of np.repeat for 1D"""
    assert len(a.size()) == 1
    return th_flatten(torch.transpose(a.repeat(repeats, 1), 0, 1))
# 在y轴上重复，然后再让x和y轴交换

def np_repeat_2d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""

    assert len(a.shape) == 2
    a = np.expand_dims(a, 0)
    a = np.tile(a, [repeats, 1, 1])
    return a

# index_select有点像map,就是inds一般是一个tensor里面有好多数，他返回输入的tensor的第index个数据，也是一个
# 优势：结果是tensor
def th_gather_2d(input, coords):
    # 3D的coords
    inds = coords[:, 0]*input.size(1) + coords[:, 1]
    # 一对 （x,y）转化到一维上的坐标，x乘x维的长度 然后加上y的长度
    x = torch.index_select(th_flatten(input), 0, inds)
    return x.view(coords.size(0))
#   最后还原成一个图的feature

# 这是一种插值方法，下面sp是样条插值方法
def th_map_coordinates(input, coords, order=1):
    """Tensorflow verion of scipy.ndimage.map_coordinates
    Note that coords is transposed and only 2D is supported
    Parameters
    ----------
    input : tf.Tensor. shape = (sizex, sizey)
    coords : tf.Tensor. shape = (n_points, 2)
    """

    assert order == 1
    input_size = input.size(0)

    coords = torch.clamp(coords, 0, input_size - 1)
    # 保证这个坐标大于0，小于这个。 其他小的是0,大的是后面这，主要目的是保证这个值不能超出图本来的尺寸
    coords_lt = coords.floor().long()
    # 对xy下取整
    coords_rb = coords.ceil().long()
    # 对xy上取整
    coords_lb = torch.stack([coords_lt[:, 0], coords_rb[:, 1]], 1)
    # 组合成x下y上
    coords_rt = torch.stack([coords_rb[:, 0], coords_lt[:, 1]], 1)
    # x上y下
    # 上面求到的这几个量是用来计算的，不可以进行grad下降，所以下面有的detach()
    vals_lt = th_gather_2d(input,  coords_lt.detach())
    # 双下
    vals_rb = th_gather_2d(input,  coords_rb.detach())
    # x 上 y上
    vals_lb = th_gather_2d(input,  coords_lb.detach())
    # x下 y上
    vals_rt = th_gather_2d(input,  coords_rt.detach())
    # x上 y下
    coords_offset_lt = coords - coords_lt.type(coords.data.type())
    # 坐标减去向下整的坐标，就得到小数坐标，然后用三次线性插值的方法，得到最后结果
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    # 对x
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    # 对y
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]
    # 对上面两个方向的结果一起
    return mapped_vals
# 返回的是一个tensor

# 输入整个图，对坐标进行修改后，直接输出结果，这里用的nearest方法
def sp_batch_map_coordinates(inputs, coords):
    """Reference implementation for batch_map_coordinates"""
    # coords = coords.clip(0, inputs.shape[1] - 1)

    assert (coords.shape[2] == 2)
    # 这说明coords他预测出来是3D的  是由两个个2D图放在一起得到的，一个是x的 一个是y的
    height = coords[:,:,0].clip(0, inputs.shape[1] - 1)
    # clip是说取0到inputs.shape[1] - 1之间的数  不能让你的offset出的步长，超过了整个图input的范围，
    width = coords[:,:,1].clip(0, inputs.shape[2] - 1)
    np.concatenate((np.expand_dims(height, axis=2), np.expand_dims(width, axis=2)), 2)

    mapped_vals = np.array([
        sp_map_coordinates(input, coord.T, mode='nearest', order=1)
        for input, coord in zip(inputs, coords)
    ])
    #     使用线性的方法找到整数位置，直接输出array但是他的输入是tensor，
    return mapped_vals

# 为什么要用batch,因为如果不用batch对每一个2D图的offset坐标，他可能重复，只能batch那一维上用不同的数字来进行区别
def th_batch_map_coordinates(input, coords, order=1):
    """Batch version of th_map_coordinates
    Only supports 2D feature maps
    Parameters
    ----------
    input : tf.Tensor. shape = (batch, height, width)
    coords : tf.Tensor. shape = (batch, n_points, 2(x and  y))
    Returns
    -------
    tf.Tensor. shape = (batchs, height, width)
    """

    batch_size = input.size(0)
    input_height = input.size(1)
    input_width = input.size(2)

    n_coords = coords.size(1)

    # coords = torch.clamp(coords, 0, input_size - 1)
    # 保证coords的值不会超出原图的尺寸,例第一个narrow，  是指把第二维轴，的0到1个位置保留，不包括1
    coords = torch.cat((torch.clamp(coords.narrow(2, 0, 1), 0, input_height - 1), torch.clamp(coords.narrow(2, 1, 1), 0, input_width - 1)), 2)

    assert (coords.size(1) == n_coords)

    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], 2)
    # x下，y上
    coords_rt = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], 2)
    # x下y上
    idx = th_repeat(torch.arange(0, batch_size), n_coords).long()
    # 主要是加一个batch维度，下面计算坐标在batch中的位置，
    idx = Variable(idx, requires_grad=False)
    if input.is_cuda:
        idx = idx.cuda()
    # 在这里定义一个函数很奇怪,但是他在这里还没调用,下面才是真正的调用
    def _get_vals_by_coords(input, coords):
        indices = torch.stack([
            idx, th_flatten(coords[..., 0]), th_flatten(coords[..., 1])
        ], 1)
        inds = indices[:, 0]*input.size(1)*input.size(2)+ indices[:, 1]*input.size(2) + indices[:, 2]
        vals = th_flatten(input).index_select(0, inds)
        vals = vals.view(batch_size, n_coords)
        return vals
    # 分别计算各个的值,
    vals_lt = _get_vals_by_coords(input, coords_lt.detach())
    vals_rb = _get_vals_by_coords(input, coords_rb.detach())
    vals_lb = _get_vals_by_coords(input, coords_lb.detach())
    vals_rt = _get_vals_by_coords(input, coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())
    # 变成小数,
    vals_t = coords_offset_lt[..., 0]*(vals_rt - vals_lt) + vals_lt
    # x方向
    vals_b = coords_offset_lt[..., 0]*(vals_rb - vals_lb) + vals_lb
    # y方向
    mapped_vals = coords_offset_lt[..., 1]* (vals_b - vals_t) + vals_t
    # 两个方向
    return mapped_vals


def sp_batch_map_offsets(input, offsets):
    """Reference implementation for tf_batch_map_offsets"""

    batch_size = input.shape[0]
    input_height = input.shape[1]
    input_width = input.shape[2]

    offsets = offsets.reshape(batch_size, -1, 2)
    grid = np.stack(np.mgrid[:input_height, :input_width], -1).reshape(-1, 2)
    grid = np.repeat([grid], batch_size, axis=0)
    # 上面两步是生成两个基数的表格，然后把神网络层跑出来的结果加到那上面，就能让不同图的offset在不同的batch上
    coords = offsets + grid
    # coords = coords.clip(0, input_size - 1)

    mapped_vals = sp_batch_map_coordinates(input, coords)
    return mapped_vals


def th_generate_grid(batch_size, input_height, input_width, dtype, cuda):
    grid = np.meshgrid(
        range(input_height), range(input_width), indexing='ij'
    )
    grid = np.stack(grid, axis=-1)
    grid = grid.reshape(-1, 2)

    grid = np_repeat_2d(grid, batch_size)
    grid = torch.from_numpy(grid).type(dtype)
    if cuda:
        grid = grid.cuda()
    return Variable(grid, requires_grad=False)


def th_batch_map_offsets(input, offsets, grid=None, order=1):
    """Batch map offsets into input
    Parameters
    ---------
    input : torch.Tensor. shape = (b, s, s)
    offsets: torch.Tensor. shape = (b, s, s, 2)
    Returns
    -------
    torch.Tensor. shape = (b, s, s)
    """
    batch_size = input.size(0)
    input_height = input.size(1)
    input_width = input.size(2)

    offsets = offsets.view(batch_size, -1, 2)
    if grid is None:
        grid = th_generate_grid(batch_size, input_height, input_width, offsets.data.type(), offsets.data.is_cuda)

    coords = offsets + grid

    mapped_vals = th_batch_map_coordinates(input, coords)
    return mapped_vals
