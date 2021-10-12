

import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from optparse import OptionParser
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from PIL import Image
import matplotlib.pyplot as plt
from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
nn.L1Loss()
def train_net(net,
              epochs=5,
              batch_size=2,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5):
        ##用默认的形参来导入函数
    # dir_img = r'/home1/hli/U-net/total_gradientxy/'  # 程序中写文件路径是都是直接dir_img+name 不考虑/的  所以这提前写好
    # dir_mask = r'/home1/hli/U-net/bone_gradientxy/'
    # dir_lung = r'/home1/hli/U-net/mask_for_drr_gradient/'
    # dir_checkpoint = r'/home1/hli/U-net/5ndlung_checkpoints/'
    # loss_txt_dir = r'/home1/hli/U-net/batch4_lr0.001.txt'
    # dir_img = r'/home/hli/U-net/total_drr_png/'  # 程序中写文件路径是都是直接dir_img+name 不考虑/的  所以这提前写好
    # dir_mask = r'/home/hli/U-net/bone_drr_png/'
    # dir_lung = r'/home/hli/U-net/mask_for_drr_gradient/'
    # dir_checkpoint = r'/home/hli/U-net/nogra_checkpoints/nomask/'
    # loss_txt_dir = r'/home/hli/U-net/loss2without_mask.txt'

#
        ##输入地扯.不load就是保存的地方 load 就是导入的

    #
    dir_img = r'/home1/hli/U-net/total_gradientxy/'  # 程序中写文件路径是都是直接dir_img+name 不考虑/的  所以这提前写好
    dir_mask = r'/home1/hli/U-net/bone_gradientxy/'
    dir_lung = r'/home1/hli/U-net/mask_for_drr_gradient/'
    dir_checkpoint = r'/home1/hli/U-net/5ndlung_checkpoints/'
    loss_txt_dir = r'/home1/hli/U-net/batch4_lr0.001.txt'


    ids = get_ids(dir_lung)##得到.png 前面的东西，但是此时ids 是一个生成器，不是一个表什么的
    ids = split_ids(ids)##打成元组的样子，比如（a，0）第0个图的id是a 也是一个生成器

    iddataset = split_train_val(ids, val_percent)
    # 返回是字典，分成训练集和验证集{'train': dataset[:-n], 'val': dataset[-n:]}

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])##刚才分出来的小片

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()##交叉熵loss
    criterion1 = nn.L1Loss()##绝对值loss
    criterion2 = nn.MSELoss(size_average=True)##均方差loss

    for epoch in range(epochs):
        time1=time.time()
        f = open(loss_txt_dir, mode='a')
        f.write('epoch:'+str(epoch)+'   time:'+str(time1)+'\n\n\n')
        f.close()
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        ##进行的次数
        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, dir_lung,img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, dir_lung,img_scale)
        ##一个image一个mask，依次放，而且都是方的
        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            ##顺便就给batch给编号了，因为train是带mask和image的，所以enumerate后是二个坐标
            ##batch_size是指的张数

            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b]).astype(np.float32)
            if (i*batch_size)%1000==0:
                plt.imshow(imgs[0,0,:,:],cmap='gray')
                # plt.show()
                # plt.imshow(true_masks[0,:,:],cmap='gray')
                # plt.show()
            mask=(1-true_masks).copy()
            mask=torch.from_numpy(mask)
            imgs = torch.from_numpy(imgs)##改成了torch的格式
            true_masks = torch.from_numpy(true_masks)

            if gpu:##放到gpu
                imgs = imgs.cuda()###
                true_masks = true_masks.cuda()
                mask=mask.cuda()

            masks_pred = net(imgs)##用网络跑出来mask

            if (i*batch_size)%1000==0:
                image=masks_pred.detach()
                image=image.cpu().numpy()

               # print(image)
                #plt.imsave('/home/hli/U-net/checkpoints/'+str(i)+'.png',image[0,0,:,:])
                #time.sleep(5)
                plt.imshow(image[0,0,:,:],cmap='gray')
                plt.show()
                plt.imshow(image[0,1,:,:],cmap='gray')
                plt.show()
            lung_masks_probs=masks_pred[:,1,:,:]
            bone_probs=masks_pred[:,0,:,:]
            lung_masks_probs_flat = F.sigmoid(lung_masks_probs.contiguous().view(-1))##pytorch中的压缩一维
            bone_probs_flat=bone_probs.contiguous().view(-1)
            true_lung_masks_flat = true_masks[:,1,:,:].contiguous().view(-1)
            true_bone_flat=true_masks[:,0,:,:].contiguous().view(-1)
            # mask=mask.view(-1)
           # print(true_masks_flat)
           #  masked_GT=mask*true_masks_flat
            # masked_pre=mask*masks_probs_flat
            loss1 = criterion1(bone_probs_flat, true_bone_flat)
            loss2=criterion2(lung_masks_probs_flat,true_lung_masks_flat)
            loss=loss1+loss2

            #做了修改，改成loss1是预测骨，loss2是预测lung mask.
            ###前面写的是BCEloss，他求的是这个两个值之间的相对熵
            #要区分下面的dice loss
            epoch_loss += loss.item()
            print('e:%s'%epoch)
            print('{0:.4f} --- loss1: {1:.6f}'.format(i * batch_size / N_train, loss1.item()))
            print('{0:.4f} --- loss2: {1:.6f}'.format(i * batch_size / N_train, loss2.item()))

            # print('epoch:{1:.0f} ---{0:.4f} --- loss1: {1:.6f}'.format(epoch,i * batch_size / N_train, loss.item()))
            # print('{0:.4f} --- loss2: {1:.6f}'.format(i * batch_size / N_train, loss2.item()))

            optimizer.zero_grad()##归0
            loss.backward()##开始降低loss
            optimizer.step()##开始步进
        f = open(loss_txt_dir, mode='a')
        f.write('loss:' + str(epoch_loss/ i))
        f.close()
        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))
        f = open(loss_txt_dir, mode='a')
        time2=time.time()
        epcho_time=time2-time1
        f.write('\n\n time2:' + str(time2) + '  epcho time:'+str(epcho_time)+'\n\n\n')
        f.close()
        # if 1:
        #     val_dice = eval_net(net, val, gpu)
        #     #这里的gpu 虽然默认是F ，但是 main中又改成了T
        #     f = open(loss_txt_dir, mode='a')
        #     f.write('Dice Coeff: '+val_dice+'\n\n\n')
        #     f.close()
        #     print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('-d','--no_deform',dest='no_deform',default=1,type='int',help='1for do not use deform 0 for use deform' )
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()  
    # torch.cuda.set_device(0)
    net=torch.nn.DataParallel(UNet(n_channels=3, n_classes=2,no_deform=0),device_ids=[0,1])
    # net = UNet(n_channels=3, n_classes=1,no_deform=0)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
