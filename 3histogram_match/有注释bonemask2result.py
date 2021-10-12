
import numpy as np
from skimage import io,morphology
from skimage import transform as T
from matplotlib import pyplot as plt
from PIL import Image
import cv2
# from poissonblending import  *
import os
import math

def Equal_Hist(source,maxvalue=255):
    oldshape = source.shape
    source = source.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    s_quantiles = np.cumsum(s_counts[1:]).astype(np.float64)#不是从0开始计的，是为
    #了不计入值为0的点。
    s_quantiles /= s_quantiles[-1] #转化成百分比
    t_quantiles=np.ones(maxvalue+1)
    t_quantiles[0]=0
    t_quantiles=np.cumsum(t_quantiles).astype(np.float64)
    t_values=np.ones(maxvalue)*0.2
    t_values=np.cumsum(t_values).astype(np.float64)
    b=np.array([10])
    t_values=np.concatenate((b,t_values+10))
    t_quantiles/=maxvalue
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    a = np.asarray([0])  # 之间没有把第一个值去均衡，现在补上
    interp_t_values = np.concatenate((a, interp_t_values))
    result=interp_t_values[bin_idx].reshape(oldshape)
    return result
def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    plt.hist(template[np.where(template>0)],normed=1,bins=140,facecolor='r')
    # 显示横轴标签
    plt.xlabel("")
    # 显示纵轴标签
    plt.ylabel("")
    # 显示图标题
    plt.title("")
    # plt.show()
    # get the set of unique pixel values and their corresponding indices and
    # counts1
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # 注意这都是在一维上的，不是二维图上take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts[1:]).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    mid=int(len(t_counts)/2)

    t_counts=t_counts[(np.where(t_values<39)) and (np.where(t_values>29))]
    t_values=t_values[(np.where(t_values<39)) and (np.where(t_values>29))]

    t_quantiles = np.cumsum(t_counts[:]).astype(np.float64)

    # t_quantiles = np.cumsum(t_counts[10:-40]).astype(np.float64)
    try:t_quantiles /= t_quantiles[-1]
    except IndexError as ID:
        print(f)
        return np.zeros(oldshape)
    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    #interp_t_values = np.zeros_like(source,dtype=float)
    # interp_t_values = np.interp(s_quantiles, t_quantiles, t_values[10:-40])
    t_values=t_values[:]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    a=np.asarray([0])#之间没有把第一个值去均衡，现在补上
    interp_t_values = np.concatenate((a, interp_t_values))

    return interp_t_values[bin_idx].reshape(oldshape)

def bonesuppress(root,xray_root,lung_root,save_root,resultsize,xraychanles,equal_hist=0):
    bone=io.imread(root)

    # bone=bone[:,:,0]
    xray_img=io.imread(xray_root)

    lung_mask=io.imread(lung_root)
    # lung_mask=lung_mask[:,:,2]
    # plt.imshow(lung_mask)
    # plt.show()
    lung_mask[np.where(lung_mask==0)]=0
    lung_mask[np.where(lung_mask==255)]=1
    times=int(math.pow((resultsize/1024),2))
    # bone[np.where(bone<60)]=0
    # bone[np.where(bone>60)]=255
    xray_shape=xray_img.shape[-1]
    if not(xray_shape == 1024):
        xray_img=xray_img[:,:,0]
    bone=T.resize(bone,(resultsize,resultsize))
    lung_mask=T.resize(lung_mask,(resultsize,resultsize))
    lung_mask=np.where(lung_mask>0,1,0)
    bone=bone*255
    bone=bone.astype(np.uint8)
    bone=bone*lung_mask

    plt.imshow(bone,cmap='gray')
    # plt.show()
    mean=np.mean(bone[np.where(lung_mask==1)])
    bone[np.where(bone<mean)]=0
    # plt.imshow(bone)
    # plt.show()
    bone_mask=np.zeros([resultsize,resultsize])
    bone_mask[np.where(bone>0)]=1
    # bone_mask=bone_mask.astype(np.bool)
    dia_bone_mask=morphology.dilation(bone_mask,np.ones([20,20])) #1024时是20 2048 40
    dia_bone_mask=dia_bone_mask*lung_mask
    # dia_bone_mask=dia_bone_mask.astype(np.bool)
    around_bone_img=xray_img*(dia_bone_mask-bone_mask)
    # plt.imshow(around_bone_img,cmap='gray')
    # plt.show()
    count=np.count_nonzero(dia_bone_mask-bone_mask)
    count2=np.count_nonzero(bone_mask)
    around_bone_mean=np.sum(around_bone_img)/count
    bone_mean=np.sum(xray_img*bone_mask)/count2
    bone_img=xray_img*bone_mask
    # plt.imshow(bone_img,cmap='gray')
    # plt.show()
    bone_img=bone_img-around_bone_mean
    bone_img[np.where(bone_img<0)]=0
    bone_img[np.where(bone_img>200)]=0
    # plt.imshow(bone_img,cmap='gray')
    # plt.show()
    mean_bone_img=np.mean(bone_img[np.where(bone_mask==1)])
    mean_match=0
    if mean_match:
        bone2match=np.array((-mean_bone_img+5,-mean_bone_img+4,0,0,0,0,0,0,1,2,3,4,5,-1,-2,-3,-4,-5))+mean_bone_img
        matched_bone=hist_match(bone,bone2match)
    else:
        matched_bone=hist_match(bone,bone_img[np.where(bone_mask==1)])
        plt.hist(matched_bone[np.where(matched_bone > 0)], normed=1, bins=140, facecolor='b')
        plt.xlabel("")
        # 显示纵轴标签
        plt.ylabel("")
        # 显示图标题
        plt.title("")
        # plt.show()
    # 虽然我的输入是bone的原图，但是我在进行match时，我把第一个值，就是背景全是0的那个0给去掉了，
    # 不match 他
    blur_matched_bone= cv2.GaussianBlur(matched_bone, (5*times, 5*times), 0)
    matched_bone=matched_bone
    blur_matched_bone=blur_matched_bone

    # blur_matched_bone=morphology.remove_small_holes(blur_matched_bone,30)
    # blur_matched_bone=morphology.dilation(blur_matched_bone,np.ones([3,3]))
    # 因为发现骨的位置普遍要小一点，所以应该放大一点
    plt.imshow(matched_bone,cmap='gray')
    # plt.show()

    # img = morphology.remove_small_objects(ero_bone.astype(np.bool), 5000)
    # img=morphology.dilation(bone,np.ones([3,3]))
    # ero_bone=morphology.erosion(img,np.ones([3,3]))

    # img = morphology.remove_small_holes(img.astype(np.bool), 20)
    # img = morphology.remove_small_objects(img.astype(np.bool), 500)
    match_hist=1
    if match_hist:
        result=xray_img-0.4*blur_matched_bone
    else:
        dia_bone_mask=morphology.dilation(bone_mask.astype(np.bool),np.ones([3,3]))#1024时是5
        dia_bone_mask=morphology.remove_small_holes(dia_bone_mask,500*times)

        result=xray_img-dia_bone_mask*mean_bone_img
    blur_border=0
    if blur_border:
        # plt.imshow(blur_matched_bone,cmap='gray')
        # plt.show()
        # 0.85最好
        blend_bone=result*bone_mask
        # plt.imshow(result,cmap='gray')
        # plt.show()
        # bone_mask=morphology.dilation(bone_mask,np.ones([2,2]))
        dia_bone_mask2blur=morphology.dilation(bone_mask.astype(np.bool),np.ones([10,10]))#1024时是5
        dia_bone_mask2blur=morphology.remove_small_holes(dia_bone_mask2blur,500*times)
        ero_bone_mask2blur=morphology.erosion(dia_bone_mask2blur,np.ones([20,20]))#1024时是10
        mask2blur=dia_bone_mask2blur^ero_bone_mask2blur
        # 上面等到是要blur的位置
        # plt.imshow(mask2blur)
        # ero_mask2blur=morphology.erosion(mask2blur,np.ones([2,2]))
        mask_not_blur=1^mask2blur


        result2blur=result.copy()
        result2blur=cv2.GaussianBlur(result,(5,5),1)
        result2blur=result2blur*mask2blur
        result_not_blur=result*mask_not_blur
        # result = blend(result, blend_bone, bone_mask, offset=(0, 0))
        result=result2blur+result_not_blur
        # blackholes=np.zeros((1024,1024))
        # blackholes[np.where(result==0)]=1

        # plt.imshow(blackholes, cmap='gray')
        # plt.show()
    if equal_hist:
        # equalhist_img=result*lung_mask
        result= Equal_Hist(result*lung_mask)
    result=result.astype(np.uint8)
    io.imsave(save_root,result)
    # plt.imshow(result,cmap='gray')

    # plt.imshow(xray_img,cmap='gray')
    # plt.show()



if __name__=='__main__':
    bone_root=r'../2Pytorch-UNet-master_trainbone&mask/bone'
    xray_root=r'./dataset'
    lung_root=r'../2Pytorch-UNet-master_trainbone&mask/lung'
    save_root=r'./result'
    # bone_root=r'C:\Users\lenovo\Desktop\py用\学长课题\improvement\testpy\gradall\bonegradall'
    # xray_root=r'F:\data\下载的原始数据\ChinaSet_AllFiles\ChinaSet_AllFiles\CXR1024size3c'
    # lung_root=r'C:\Users\lenovo\Desktop\py用\学长课题\improvement\testpy\gradall\lunggradall'
    # save_root=r'F:\data\new_bone_suppress_result7_noblurborder'
    size=1024#定义一个size ，就是想让这个输出图是多少的尺寸
    xray_root_list=os.listdir(xray_root)
    xray_root_list.sort()
    count=0
    for f in xray_root_list:
        count+=1
        result_f,_=f.split('.')
        result_f=result_f+'_fake_B.png'
        xray_name=os.path.join(xray_root,f)
        bone_name=os.path.join(bone_root,result_f)
        lung_name=os.path.join(lung_root,result_f)
        save_name=os.path.join(save_root,f)
        bonesuppress(bone_name, xray_name, lung_name, save_name,size,xraychanles=1,equal_hist=0)
        print(f)
        print(count)

    # f='CHNCXR_0171_0.png'
    # result_f,_=f.split('.')
    # result_f=result_f+'_fake_B.png'
    # xray_name=os.path.join(xray_root,f)
    # bone_name=os.path.join(bone_root,result_f)
    # lung_name=os.path.join(lung_root,result_f)
    # save_name=os.path.join(save_root,f)
    # bonesuppress(bone_name, xray_name, lung_name, save_name)