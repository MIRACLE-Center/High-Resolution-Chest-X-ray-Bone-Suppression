import argparse
import os
import sys
from eval import  eval_net
import numpy as np
import torch
import torch.nn.functional as F
from skimage import transform as T
from PIL import Image

from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks, dense_crf
from utils import plot_img_and_mask

from torchvision import transforms
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=False):

    # img_height = full_img.size[1]
    # img_width = full_img.size[0]

    # img = resize_and_crop(full_img, scale=scale_factor)
    # img = normalize(img)
    img=full_img
    # left_square, right_square = split_img_into_squares(img)

    img = hwc_to_chw(img)
    img = np.expand_dims(img, 0)

    # right_square = hwc_to_chw(right_square)
    #坐标转化
    # img = torch.from_numpy(img).unsqueeze(0)
    img = torch.from_numpy(img)

    # X_right = torch.from_numpy(right_square).unsqueeze(0)
    #都要压缩到第一轴上
    if use_gpu:
        img = img.cuda()
        # X_right = X_right.cuda()

    with torch.no_grad():##反正是个上下文，会自己退出
        result = net(img)##用网络直接
        # output_right = net(X_right)

        # left_probs = torch.sigmoid(output_left).squeeze(0)
        # right_probs = torch.sigmoid(output_right).squeeze(0)
        ##使用sigmoid 计算
        # tf = transforms.Compose(
        #     [
                # transforms.ToPILImage(),
                # transforms.Resize(img_height),
                # transforms.ToTensor()
            # ]
        # )
        ##这个函数可以把好多属性放在一起
        # left_probs = tf(left_probs.cpu())
        # right_probs = tf(right_probs.cpu())

        result = result.squeeze().cpu().numpy()
        ##.cpu是因为他本来在gpu内。把他弄到cpu
        # right_mask_np = right_probs.squeeze().cpu().numpy()

    # full_mask = merge_masks(left_mask_np, right_mask_np, img_width)

    # if use_dense_crf:
    #     full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)

    return result #> out_threshold##直接用>号判断他是大于阈值的部分输出
    ##一般不用语义分割


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default=r'C:\Users\lenovo\Desktop\CP9.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
                        ##输入训练的状态文件
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=False)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
                        ##nargs 表示参数个数，这里的+表示有一个或多个参数
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=True)##是否要显示图
                    ##action 表示有东西输入就存为true  这默认是false
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=1)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()
    # 运行的格式：
    # Python　predict.py
    # -m "/home/LH/data/checkpoints/CP5.pth"
    # -i "/home/LH/data/train/0cdf5b5d0ce1_01.jpg"
    # -n -v
def get_output_filenames(args):##生成输出文件
    in_files = args.input
    out_files = []##用的列表的形式
    # 应该是指定输出文件的全路径名称，否则他会自己做，在输入文件名前加上_OUT
    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)##提出他的上级路径
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        # 输入和输出的list自己写的时候 要对应好
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()##可以让sys退出
    else:
        out_files = args.output
        ##如果指定的文件，就直接输出就可以

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

if __name__ == "__main__":
    args = get_args()##刚才写的参数的函数全都可以用了
    # in_files = args.input
    in_files=[r'/home/hli/U-net/train_total_xy/255xy/34deep_xy.png']
    # out_files = get_output_filenames(args)##上面的生成文件的函数

    net = torch.nn.DataParallel(UNet(n_channels=3, n_classes=1),device_ids=[0])
    ##一张图3个通道，
    print("Loading model {}".format(args.model))
    sys.stdout.flush()
    if not args.cpu:##选择用什么来跑数据
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))

    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))
        sys.stdout.flush()
        img = Image.open(fn)
        img=np.array(img,dtype=np.float32)
        img=(img-np.min(img))/(np.max(img)-np.min(img))
        # img=np.resize(img,(384,384))
        # # 因为之前的Xray的gradient 是1通道，大尺寸的，所以要进行这几个操作
        img=np.stack((img,img,img),2)
        # if img.size[0] < img.size[1]:
        #     print("Error: image height larger than the width")

        sys.stdout.flush()
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,##输入口
                           out_threshold=args.mask_threshold,##再从输入口弄一个阈值
                           use_dense_crf= not args.no_crf,
                           use_gpu=not args.cpu)

        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)#这里的mask可以不是255的整数，因为我们不是保存

        # if not args.no_save:##如果要保存
        #     out_fn = out_files[i]
        #     result = mask_to_image(mask)
        #     result.save(out_files[i])
        #
        #     print("Mask saved to {}".format(out_files[i]))
        #     sys.stdout.flush()