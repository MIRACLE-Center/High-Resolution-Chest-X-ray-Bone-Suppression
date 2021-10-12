import random
import numpy as np


def get_square(img, pos):
    ##这个img是2D的
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]
    ##负数是从倒数第几个开始的，这个相当于把这个图的x轴对调了
    ##所以他是左边生成一个或右边生成了一个下方形
def split_img_into_squares(img):##返回左右两个方形
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    # img=np.concatenate(i)
    return np.transpose(img, axes=[2, 0, 1])
##第3轴不变，再关于第1和0对角线T，完成后，对图转一下发现就是正常的坐标顺序
def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]##要被split的图的尺寸
    h = pilimg.size[1]
    newW = int(w * scale)##缩放后的尺寸
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    #根据（左，底，右，顶）的顺序写的，且不包括右上限
    return np.array(img, dtype=np.float32)

def batch(iterable, batch_size):
    """Yields lists by batch其实就是化成好多池"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        # 根据batch_size来确定把几个zip打成一个list 然后输出
        if (i + 1) % batch_size == 0:
            ##+1是因为从0开始的
            yield b##是一个generator 迭代器。
            b = []

    if len(b) > 0:
        ##这是防止他还会有剩下的b没有达到尺寸，可以直接弄出来
        yield b

def split_train_val(dataset, val_percent=0.05):
    ##把训练集切成小段
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)##主要是看取多少个小段
    random.shuffle(dataset)##乱序
    return {'train': dataset[:-n], 'val': dataset[-n:]}
    #从最后一个到倒数第n个，做训练集，以及 从倒n到最后一个做验证集
# 在一个训练中他们是不变的，并不是一个epoch变一次
def normalize(x):
    return x / 255

def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]
    #直接用一个空的接收
    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs
