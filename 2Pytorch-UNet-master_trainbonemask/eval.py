import torch
import torch.nn.functional as F

from dice_loss import dice_coeff

##evaluation
def eval_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    ##用coefficient 来验证他mask的准确性
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0]
        #2D,因为dataset是3D的，而且是一个pre的一个是ground truth
        true_mask = b[1]

        img = torch.from_numpy(img).unsqueeze(0)
        ##生成一个tensor,否则不能跑
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0,:,:,:]

        lung_pred = (F.sigmoid(mask_pred[1,:,:]) > 0.5).float()
        bone_pred = mask_pred[0,:,:].float()
        tot += dice_coeff(lung_pred, true_mask[1,:,:]).item()
        tot+=  dice_coeff(bone_pred,true_mask[0,:,:]).item()
        return tot / (i+1)#取平均
