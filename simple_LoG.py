import numpy as np
import cv2
import SimpleITK as sitk
import skimage.transform as T
import matplotlib.pyplot as plt
from skimage import io,exposure


def img2gradietn(img,filterxx,filteryy):
    plt.imshow(img,cmap='gray')
    plt.show()
    img = (img - np.min(img))/ (np.max(img) - np.min(img))
    img = img * 255
    img.astype(np.float64)


    img = T.resize(img, (384, 384))

    filterxxyy=filterxx+filteryy
    img_x = cv2.filter2D(img, -1, filterxx)
    img_y = cv2.filter2D(img, -1, filteryy)
    # img_x_and_y = cv2.filter2D(img, -1, filterxxyy)
    img_x = np.abs(img_x)#他有正有负的
    img_y = np.abs(img_y)
    img_x = exposure.adjust_gamma(img_x, 0.5)
    img_y = exposure.adjust_gamma(img_y, 0.5)

    img_x_and_y = np.abs(img_x + img_y) #use  filterxxyy directly will get the same result

    img_x_and_y = (img_x_and_y - np.min(img_x_and_y))  / (np.max(img_x_and_y) - np.min(img_x_and_y))

    img_x = (img_x - np.min(img_x))/ (np.max(img_x) - np.min(img_x))
    img_y = (img_y - np.min(img_y))/ (np.max(img_y) - np.min(img_y))

    return  img_x.astype(np.float32),img_y.astype(np.float32),img_x_and_y.astype(np.float32)

if __name__ == '__main__':
    filter_x=np.zeros((3,3),np.float64)
    filter_y=filter_x.copy()
    filterxx = np.zeros((5, 5), np.float64)
    filteryy = filterxx.copy()
    PI=np.pi
    W = 2
    sigma = 1
    for i in range(-2, 3): #5*5
        for j in range(-2, 3):
            filterxx[i + W, j + W] = (1 - (i * i) / (sigma * sigma)) * np.exp(
                -1 * (i * i + j * j) / (2 * sigma * sigma)) * (-1 / (2 * PI * pow(sigma, 4)))
            filteryy[i + W, j + W] = (1 - (j * j) / (sigma * sigma)) * np.exp(
                -1 * (i * i + j * j) / (2 * sigma * sigma)) * (-1 / (2 * PI * pow(sigma, 4)))
    # cv2.Laplacian()
    # for i in range(-1, 2):#3*3
    #     for j in range(-1, 2):
    #         filterxx[i + W, j + W] = (1 - (i * i) / (sigma * sigma)) * np.exp(
    #             -1 * (i * i + j * j) / (2 * sigma * sigma)) * (-1 / (2 * PI * pow(sigma, 4)))
    #         filteryy[i + W, j + W] = (1 - (j * j) / (sigma * sigma)) * np.exp(
    #             -1 * (i * i + j * j) / (2 * sigma * sigma)) * (-1 / (2 * PI * pow(sigma, 4)))

    # print(filterxx)
    # print(filteryy)
    z=filterxx+filteryy
    print(z)


    file_name=r'.\3histogram_match\dataset\CHNCXR_0002_0.png' #your file name
    total_img=io.imread(file_name)
    # total_img=total_img[0,:,:]



    total_x,total_y,total_xy=img2gradietn(total_img,filterxx,filteryy)


    plt.imshow(total_xy,cmap='gray')
    plt.show()
    # plt.imshow(total_x,cmap='gray')
    # plt.show()
    # plt.imshow(total_y,cmap='gray')
    # plt.show()



