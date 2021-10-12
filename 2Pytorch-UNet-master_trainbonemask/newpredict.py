

import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from optparse import OptionParser
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from unet import UNet
from skimage import io,morphology

def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s



def predict_net(net,allfilename,gpu=False):

    imgs = Image.open(allfilename)
    imgs = np.array(imgs, dtype=np.float32)
    # imgs = np.stack((imgs, imgs, imgs), 2)
    imgs = imgs.transpose((2, 0, 1))
    imgs = np.expand_dims(imgs, 0)
    imgs = torch.from_numpy(imgs)
    if gpu:##放到gpu
        imgs = imgs.cuda()###
    masks_pred = net(imgs)##用网络跑出来mask
    image=masks_pred.detach()
    image=image.cpu().numpy()
    #plt.imshow(image[0,0,:,:],cmap='gray')
   # plt.show()
    image[np.where(image<0)]=0
    bone=image[0,0,:,:]*255
    lung_mask=image[0,1,:,:]
    lung_mask=sigmoid(lung_mask)
    lung_mask_mean=np.mean(lung_mask)
    lung_mask[np.where(lung_mask>lung_mask_mean)]=1
    lung_mask[np.where(lung_mask<1)]=0
    lung_mask=morphology.remove_small_objects(lung_mask.astype(np.bool),500)
    # plt.imshow(bone,cmap='gray')
    # plt.show()
    # plt.imshow(lung_mask,cmap='gray')
    # plt.show()

    bone=bone.astype(np.uint8)
    lung_mask=lung_mask.astype(np.uint8)
    lung_mask=lung_mask*255
    _,filename=allfilename.split(root+'/')

    save_filename=os.path.join(save_root,filename)
    save_lung_filename=os.path.join(save_lung,filename)
    # plt.imshow(bone,cmap='gray')
    # plt.show()
    io.imsave(save_filename,bone)
    io.imsave(save_lung_filename,lung_mask)



def get_args():
    parser = OptionParser()
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='path to load file model')
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    root=r'./dataset'
    save_root=r'./bone'
    save_lung=r'./lung'#11wXray的时候写反了
    args = get_args()
    # torch.cuda.set_device(0)
    net=torch.nn.DataParallel(UNet(n_channels=3, n_classes=2,no_deform=0),device_ids=[0,1,2])
    # net = UNet(n_channels=3, n_classes=1,no_deform=0)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory
        for f in os.listdir(root):
            filename=os.path.join(root,f)
            print(f)
            predict_net(net=net,allfilename=filename,gpu=args.gpu)