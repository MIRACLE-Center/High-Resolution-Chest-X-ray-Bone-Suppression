#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[0:-11] for f in os.listdir(dir))
##因为他的文件名字造成的，为什么是-4 是因为.jpg正好是4个，他用这这种方法提出了前几位

def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)
##是给每个id n个tupels

def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        # im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        im = Image.open(dir + id + suffix)
        im= np.array(im, dtype=np.float32)
        # stack要放在yield前面，如果写到下面那个函数里，就会出现在stack时
        # 调用了三次im
        # im=(im-np.min(im))/(np.max(im)-np.min(im))
        im = np.stack((im, im, im), 2)
        # yield get_square(im, pos)##取方形
        ##一次次的进行
        yield im
def new_to_cropped_imgs(ids, dir2,dir3, suffix2, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        # im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        im2 = Image.open(dir2 + id + suffix2)
        im3 = Image.open(dir3 + id + 'deep_xy.png')

        im2= np.array(im2, dtype=np.float32)
        im3=np.array(im3,dtype=np.float32)
        im2=(im2-np.min(im2))/(np.max(im2)-np.min(im2))
        im3=(im3-np.min(im3))/(np.max(im3)-np.min(im3))
        im3=im3[:,:,1]
        # yield get_square(im, pos)##取方形
        ##一次次的进行
        result=np.stack((im2,im3),2)
        yield result
def get_imgs_and_masks(ids, dir_img, dir_mask, dir_lung,scale):
    """Return all the couples (img, mask)"""
    # 下面的crooped  其实没有进行放缩变换。只进行导入
    imgs = to_cropped_imgs(ids, dir_img, 'deep_xy.png', scale)
    # imgs=np.stack((imgs,imgs,imgs),2) deepdrr  deep_xy
    # need to transform from HWC to CHW,因为我输入的是RGB的，做lung时 是单个的,所以在上一步，我要做的一个
    # 将输入改成3通道的操作，
    imgs_switched = map(hwc_to_chw, imgs)##是把第一个作用到第二个list上，并且要返回list
    # 这是为了用torch.tensor 他的顺序是chw
    # imgs_normalized = map(normalize, imgs_switched)
    # masks=imgs_switched.copy()
    masks = new_to_cropped_imgs(ids, dir_mask,dir_lung, 'bonedrr_xy.png', scale)
    masks = map(hwc_to_chw, masks)##是把第一个作用到第二个list上，并且要返回list,bonedrr_xy bonedrr
    ##把对应的mask切成方形
    # return zip(imgs_normalized, masks)
    return zip(imgs_switched, masks)
    ##一个个对应压缩

def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)
